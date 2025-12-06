import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import multiprocessing
import random
import copy
import gc  # Added for garbage collection

# ==========================================
# CONFIGURATION (GTX 1080 SAFE)
# ==========================================
CONFIG = {
    # --- Paths ---
    "DATA_DIR": r"D:\ChirpBirdClassifier\Spectrograms_384_JPG",
    "MODEL_SAVE_DIR": r"D:\ChirpBirdClassifier\models_HighRes",

    # --- Model Dimensions ---
    "IMG_SIZE": 384,
    "BATCH_SIZE": 8,  # Strict limit for 8GB VRAM

    # --- Training Params ---
    "EPOCHS": 100,
    "LEARNING_RATE": 0.0001,
    "NUM_WORKERS": 6,

    # --- Regularization ---
    "USE_MIXUP": True,
    "MIXUP_ALPHA": 0.4,
    "NOISE_FACTOR": 0.1,  # Strength of Gaussian Noise

    # --- Hardware ---
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}


# ==========================================
# CUSTOM TRANSFORMS (NOISE)
# ==========================================
class AddGaussianNoise(object):
    """Injects random Gaussian noise for generalization."""

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # Generate noise with the same shape as the image tensor
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


# ==========================================
# MIXUP UTILS
# ==========================================
def mixup_data(x, y, alpha=0.4, device='cuda'):
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


# ==========================================
# AUGMENTATIONS
# ==========================================
class SpecAugment(object):
    def __init__(self, freq_mask=30, time_mask=40):
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def __call__(self, tensor):
        aug = tensor.clone()
        c, h, w = tensor.shape
        # Freq
        f = random.randint(0, self.freq_mask)
        f0 = random.randint(0, h - f)
        aug[:, f0:f0 + f, :] = 0
        # Time
        t = random.randint(0, self.time_mask)
        t0 = random.randint(0, w - t)
        aug[:, :, t0:t0 + t] = 0
        return aug


def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.RandomCrop(CONFIG["IMG_SIZE"], padding=8, padding_mode='reflect'),
        transforms.ToTensor(),
        # 1. Add Noise BEFORE Normalization (simulating sensor noise)
        AddGaussianNoise(0., CONFIG["NOISE_FACTOR"]),
        # 2. SpecAugment
        SpecAugment(freq_mask=30, time_mask=40),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transforms, val_transforms


# ==========================================
# DATASET (MEMORY SAFE)
# ==========================================
class BirdDataset(Dataset):
    def __init__(self, files, labels, class_to_idx, transform=None):
        self.files = files
        self.labels = labels
        self.map = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.map[self.labels[idx]]
        try:
            # SAFETY: Context manager ensures file handle closes immediately
            with Image.open(path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            # Return a blank tensor to prevent crash, but log could be added here
            return torch.zeros((3, CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])), label


# ==========================================
# TRAINING ENGINE
# ==========================================
def train_model():
    print(f"--- Starting Training on {CONFIG['DEVICE']} ---")
    os.makedirs(CONFIG["MODEL_SAVE_DIR"], exist_ok=True)

    # 1. Load Data
    print("Indexing files...")
    all_files, all_labels = [], []
    if not os.path.exists(CONFIG["DATA_DIR"]):
        print(f"ERROR: Directory {CONFIG['DATA_DIR']} not found.")
        return

    # Using scandir is slightly faster and more memory efficient for massive dirs
    classes = sorted([d.name for d in os.scandir(CONFIG["DATA_DIR"]) if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for c in classes:
        p = os.path.join(CONFIG["DATA_DIR"], c)
        # Only list specific extensions to be safe
        fs = [os.path.join(p, x) for x in os.listdir(p) if x.endswith(('.png', '.jpg', '.jpeg'))]
        all_files.extend(fs)
        all_labels.extend([c] * len(fs))

    print(f"Found {len(all_files)} images across {len(classes)} classes.")

    # 2. Split
    train_f, val_f, train_l, val_l = train_test_split(all_files, all_labels, test_size=0.15, stratify=all_labels,
                                                      random_state=42)

    # SAFETY: Clear the massive temporary lists to free RAM
    del all_files, all_labels
    gc.collect()

    # 3. Loaders
    t_trans, v_trans = get_transforms()
    train_ds = BirdDataset(train_f, train_l, class_to_idx, t_trans)
    val_ds = BirdDataset(val_f, val_l, class_to_idx, v_trans)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                              num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                            num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

    # 4. Model
    print("Initializing EfficientNet-B4...")
    weights = models.EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(CONFIG["DEVICE"])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_acc = 0.0

    # 5. Training Loop
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{CONFIG['EPOCHS']}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(CONFIG["DEVICE"]), lbls.to(CONFIG["DEVICE"])

            optimizer.zero_grad()

            apply_mixup = CONFIG["USE_MIXUP"] and (random.random() > 0.5)
            if apply_mixup:
                mixed_imgs, targets_a, targets_b, lam = mixup_data(imgs, lbls, CONFIG["MIXUP_ALPHA"], CONFIG["DEVICE"])
                outputs = model(mixed_imgs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            target_for_acc = targets_a if (apply_mixup and lam > 0.5) else lbls
            if apply_mixup and lam <= 0.5: target_for_acc = targets_b

            correct += (preds == target_for_acc).sum().item()
            total += lbls.size(0)
            pbar.set_postfix(acc=f"{correct / total:.3f}", loss=f"{total_loss / len(train_loader):.3f}")

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(CONFIG["DEVICE"]), lbls.to(CONFIG["DEVICE"])
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == lbls).sum().item()
                v_total += lbls.size(0)

        val_acc = v_correct / v_total
        print(f"   >>> Val Acc: {val_acc:.4f}")
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG["MODEL_SAVE_DIR"], "best_model_b4_mixup.pth"))
            print(f"   [!] Saved Best.")

        # SAFETY: Garbage collection and VRAM cleanup after every epoch
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_model()