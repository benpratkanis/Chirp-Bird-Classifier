const fs = require('fs');
const https = require('https');
const path = require('path');

const speciesImages = {
    "American_Crow": "https://images.unsplash.com/photo-1555169062-013468b47731?q=80&w=1000&auto=format&fit=crop",
    "American_Goldfinch": "https://images.unsplash.com/photo-1550853024-843963436971?q=80&w=1000&auto=format&fit=crop",
    "American_Robin": "https://images.unsplash.com/photo-1620662736427-b8a198f52c4d?q=80&w=1000&auto=format&fit=crop",
    "Black-capped_Chickadee": "https://images.unsplash.com/photo-1516660338335-51e60475734d?q=80&w=1000&auto=format&fit=crop",
    "Blue_Jay": "https://images.unsplash.com/photo-1549608276-5786777e6587?q=80&w=1000&auto=format&fit=crop",
    "Dark-eyed_Junco": "https://images.unsplash.com/photo-1588663823906-e306054b6702?q=80&w=1000&auto=format&fit=crop",
    "House_Finch": "https://images.unsplash.com/photo-1605092676920-1963ae35a372?q=80&w=1000&auto=format&fit=crop",
    "Northern_Cardinal": "https://images.unsplash.com/photo-1547970810-dc1eac37d174?q=80&w=1000&auto=format&fit=crop",
    "Northern_Flicker": "https://images.unsplash.com/photo-1559257606-c4238969f063?q=80&w=1000&auto=format&fit=crop",
    "Red-bellied_Woodpecker": "https://images.unsplash.com/photo-1596574202467-9e512457c38e?q=80&w=1000&auto=format&fit=crop",
    "Red-winged_Blackbird": "https://images.unsplash.com/photo-1591608971362-f08b2a75731a?q=80&w=1000&auto=format&fit=crop",
    "Song_Sparrow": "https://images.unsplash.com/photo-1589136777351-94328825c103?q=80&w=1000&auto=format&fit=crop",
    "Tufted_Titmouse": "https://images.unsplash.com/photo-1585128993285-9e3115e4351c?q=80&w=1000&auto=format&fit=crop",
    "White-breasted_Nuthatch": "https://images.unsplash.com/photo-1582268611958-ebfd161ef9cf?q=80&w=1000&auto=format&fit=crop"
};

const downloadImage = (url, filepath) => {
    return new Promise((resolve, reject) => {
        https.get(url, (res) => {
            if (res.statusCode === 200) {
                res.pipe(fs.createWriteStream(filepath))
                    .on('error', reject)
                    .once('close', () => resolve(filepath));
            } else {
                res.resume();
                reject(new Error(`Request Failed With a Status Code: ${res.statusCode}`));
            }
        });
    });
};

const outputDir = path.join(__dirname, 'frontend', 'src', 'assets', 'birds');

if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

(async () => {
    for (const [species, url] of Object.entries(speciesImages)) {
        try {
            console.log(`Downloading ${species}...`);
            await downloadImage(url, path.join(outputDir, `${species}.jpg`));
            console.log(`Downloaded ${species}`);
        } catch (e) {
            console.error(`Failed to download ${species}:`, e);
        }
    }
})();
