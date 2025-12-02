import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import multiprocessing
import copy
import random

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
CONFIG = {
    # --- Paths ---
    "DATA_DIR": r"D:\ChirpBirdClassifier\SpectrogramsCompleteSetBalanced",
    "MODEL_SAVE_DIR": r"D:\ChirpBirdClassifier\models_CompleteSet",

    # --- Data Params ---
    "IMG_SIZE": 224,
    "BATCH_SIZE": 64,
    "NUM_WORKERS": 8,

    # --- Data Splitting ---
    "SPLIT_RATIOS": {'train': 0.8, 'val': 0.1, 'test': 0.1},

    # --- Training Strategy ---
    "TRAINING_STRATEGY": 'simple',
    "N_SPLITS": 5,

    # --- Model Selection ---
    # CHANGED: Switched to ResNet34 for better stability on audio
    "MODEL_NAME": 'resnet34',

    # --- Training Params ---
    "EPOCHS": 75,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,  # Increased slightly due to MixUp

    # --- Regularization ---
    "MIXUP_ALPHA": 0.4,  # Controls how "mixed" the images get. 0.2-0.4 is standard.

    # --- Scheduler ---
    "USE_SCHEDULER": True,
    "SCHEDULER_FACTOR": 0.1,
    "SCHEDULER_PATIENCE": 3,

    # --- System ---
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}


# ==========================================
# 2. CUSTOM TRANSFORMATIONS (NOISE & SPECAUGMENT)
# ==========================================
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class SpecAugment(object):
    """
    Randomly masks out vertical (time) and horizontal (frequency) strips.
    """

    def __init__(self, freq_mask_param=20, time_mask_param=20, num_freq_masks=1, num_time_masks=1):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, tensor):
        # tensor shape: [C, H, W]
        c, h, w = tensor.shape
        augmented_spec = tensor.clone()

        # Frequency Masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, h - f)
            augmented_spec[:, f0:f0 + f, :] = 0

        # Time Masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, w - t)
            augmented_spec[:, :, t0:t0 + t] = 0

        return augmented_spec


# ==========================================
# 3. DATASET CLASS
# ==========================================
class BirdSpectrogramDataset(Dataset):
    def __init__(self, file_list, labels, class_to_idx, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        label_str = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.class_to_idx[label_str]
            return image, label
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}. Generating dummy.")
            dummy_img = torch.zeros((3, CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))
            label = self.class_to_idx[label_str]
            return dummy_img, label


# ==========================================
# 4. MODEL FACTORY
# ==========================================
def initialize_model(model_name, num_classes, device):
    print(f"Initializing {model_name}...")
    model = None

    if model_name == 'resnet34':
        weights = models.ResNet34_Weights.DEFAULT
        model = models.resnet34(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Keep robust dropout
            nn.Linear(num_ftrs, num_classes)
        )

    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model.to(device)


# ==========================================
# 5. DATA UTILS
# ==========================================
def get_all_files():
    print("Scanning dataset...")
    all_files = []
    all_labels = []

    if not os.path.exists(CONFIG["DATA_DIR"]):
        raise FileNotFoundError(f"Directory {CONFIG['DATA_DIR']} not found.")

    classes = sorted([d for d in os.listdir(CONFIG["DATA_DIR"])
                      if os.path.isdir(os.path.join(CONFIG["DATA_DIR"], d))])

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for cls_name in classes:
        cls_folder = os.path.join(CONFIG["DATA_DIR"], cls_name)
        files = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder)
                 if f.lower().endswith(('.png', '.jpg'))]
        all_files.extend(files)
        all_labels.extend([cls_name] * len(files))

    print(f"Found {len(all_files)} images across {len(classes)} classes.")
    return np.array(all_files), np.array(all_labels), class_to_idx


def get_transforms():
    # ADDED: SpecAugment is inserted after ToTensor but before Normalize
    train_transforms = transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        SpecAugment(freq_mask_param=30, time_mask_param=40),  # Cut larger holes in spectrogram
        AddGaussianNoise(0., 0.05),  # Reduced noise slightly to let MixUp do the work
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transforms, val_transforms


# ==========================================
# 6. TRAINING ENGINE WITH MIXUP
# ==========================================
def mixup_data(x, y, alpha=0.4, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_fold(train_files, train_labels, val_files, val_labels, class_to_idx, fold_name="Main",
                   save_path_override=None):
    train_transforms, val_transforms = get_transforms()
    train_ds = BirdSpectrogramDataset(train_files, train_labels, class_to_idx, train_transforms)
    val_ds = BirdSpectrogramDataset(val_files, val_labels, class_to_idx, val_transforms)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                              num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                            num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

    model = initialize_model(CONFIG["MODEL_NAME"], len(class_to_idx), CONFIG["DEVICE"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=1e-4)

    scheduler = None
    if CONFIG["USE_SCHEDULER"]:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=CONFIG["SCHEDULER_FACTOR"],
            patience=CONFIG["SCHEDULER_PATIENCE"]
        )

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if save_path_override:
        checkpoint_path = save_path_override
    else:
        checkpoint_path = os.path.join(CONFIG["MODEL_SAVE_DIR"], f"best_model_{fold_name}.pth")

    print(f"\nStarting training for: {fold_name} (MixUp Enabled)")

    try:
        for epoch in range(CONFIG["EPOCHS"]):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            pbar = tqdm(train_loader, desc=f"[{fold_name}] Ep {epoch + 1}", leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])

                optimizer.zero_grad()

                # --- MIXUP LOGIC START ---
                # Apply MixUp only during training
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, CONFIG["MIXUP_ALPHA"], CONFIG["DEVICE"])

                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                # --- MIXUP LOGIC END ---

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                # Accuracy calc is tricky with MixUp, we compare to the "dominant" label (max lambda)
                # just for logging purposes.
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                # If lam > 0.5, we compare to target_a, else target_b
                target_for_acc = targets_a if lam > 0.5 else targets_b
                correct += (predicted == target_for_acc).sum().item()

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(train_ds)
            epoch_acc = correct / total

            # Validation (Standard, No MixUp)
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_epoch_loss = val_loss / len(val_ds)
            val_epoch_acc = val_correct / val_total

            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            history['val_loss'].append(val_epoch_loss)
            history['val_acc'].append(val_epoch_acc)

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"[{fold_name} Ep {epoch + 1}] Train Acc: {epoch_acc:.4f} | Val Acc: {val_epoch_acc:.4f} | LR: {current_lr:.2e}")

            if scheduler:
                scheduler.step(val_epoch_acc)

            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"   >>> New Best Model! Saving...", end=" ", flush=True)
                torch.save(model.state_dict(), checkpoint_path)
                print("Saved!", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["PATIENCE"]:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    except KeyboardInterrupt:
        print(f"\n[!] Training interrupted! Best model (Acc: {best_acc:.4f}) is at {checkpoint_path}.")

    model.load_state_dict(best_model_wts)
    return model, history, best_acc


# ==========================================
# 7. MAIN WORKFLOW
# ==========================================
def main():
    os.makedirs(CONFIG["MODEL_SAVE_DIR"], exist_ok=True)
    all_files, all_labels, class_to_idx = get_all_files()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    torch.save(idx_to_class, os.path.join(CONFIG["MODEL_SAVE_DIR"], "class_mapping.pth"))

    test_size = CONFIG["SPLIT_RATIOS"]["test"]
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        all_files, all_labels, test_size=test_size, stratify=all_labels, random_state=42
    )

    if CONFIG["TRAINING_STRATEGY"] == 'simple':
        remaining_ratio = 1.0 - test_size
        val_ratio_adjusted = CONFIG["SPLIT_RATIOS"]["val"] / remaining_ratio

        X_train, X_val, y_train, y_val = train_test_split(
            X_remaining, y_remaining, test_size=val_ratio_adjusted, stratify=y_remaining, random_state=42
        )
        save_path = os.path.join(CONFIG["MODEL_SAVE_DIR"], "best_model_resnet34_mixup.pth")

        train_one_fold(X_train, y_train, X_val, y_val, class_to_idx, fold_name="Simple", save_path_override=save_path)

    elif CONFIG["TRAINING_STRATEGY"] == 'kfold':
        skf = StratifiedKFold(n_splits=CONFIG["N_SPLITS"], shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_remaining, y_remaining)):
            X_train_fold, y_train_fold = X_remaining[train_idx], y_remaining[train_idx]
            X_val_fold, y_val_fold = X_remaining[val_idx], y_remaining[val_idx]
            train_one_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, class_to_idx, f"Fold_{fold + 1}")


def plot_history(history, save_dir, title="History"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title(f'{title} Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title(f'{title} Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{title}.png'))
    plt.close()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()