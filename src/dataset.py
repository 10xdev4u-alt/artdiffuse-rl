"""Dataset loader for Midjourney text-to-image data.

Loads JSON files with prompts and image links, downloads images,
and returns (image, caption) pairs for training.
"""

import json
import os
import requests
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from io import BytesIO


class MidjourneyDataset(Dataset):
    """Dataset for Midjourney text-to-image data.

    Args:
        data_dir: Directory containing JSON files
        image_size: Target image size
        max_samples: Maximum number of samples to load
    """

    def __init__(self, data_dir: str, image_size: int = 64, max_samples: int = 50000):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_samples = max_samples

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

        # Load all JSON files
        self.data = []
        for json_file in sorted(self.data_dir.glob("*.json")):
            with open(json_file) as f:
                entries = json.load(f)
                for entry in entries:
                    if isinstance(entry, dict):
                        prompt = entry.get("prompt", entry.get("text", ""))
                        image_url = entry.get("image_url", entry.get("url", ""))
                        if prompt and image_url:
                            self.data.append({"prompt": prompt, "image_url": image_url})
                    if len(self.data) >= max_samples:
                        break
            if len(self.data) >= max_samples:
                break

        print(f"Loaded {len(self.data)} samples from {data_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        try:
            response = requests.get(entry["image_url"], timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            # Return a blank image on failure
            image = torch.zeros(3, self.image_size, self.image_size)

        return {"image": image, "caption": entry["prompt"]}
