# project_datasets/vsr_dataset.py
from pathlib import Path
import json
import requests
from io import BytesIO
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class VSRDataset(Dataset):
    """
    Visual Spatial Relations (VSR) Dataset
    """
    def __init__(self, dataset_name="zero_shot", split="train", base_path="project_data/raw/vsr", transform=None):

        # Validations
        
        #assert transform is not None, "Transform cannot be None. Please provide a valid transform."
        assert os.path.exists(base_path), f"Root directory '{base_path}' does not exist."
        assert split in ['train', 'dev', 'test'], f"Unsupported split: '{split}'. Must be one of ['train', 'dev', 'test']."
        
        # Img transformation
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Get train/dev/test
        self.base_path = Path(base_path) / dataset_name
        self.split = split
        self.dataset = self._load_jsonl(self.base_path / f"{split}.jsonl")

    def _load_jsonl(self, filepath):
        items = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        return items
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["caption"]
        label = item["label"]
        relation = item["relation"]
        #image_link = item["image_link"]

        #Load image
        response = requests.get(item["image_link"], timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        return {
            "image": self.transform(image),
            "text": text,
            "label": label,
            "relation": relation,
            #"image_link": image_link,
        }

def get_vsr_loader(split="train", batch_size=8, shuffle=False, transform=None):
    dataset = VSRDataset(split=split, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
