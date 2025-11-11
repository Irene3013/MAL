# project_datasets/vsr_dataset.py
from pathlib import Path
import json
import requests
from io import BytesIO
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image

class VSRDataset(Dataset):
    """
    Visual Spatial Relations (VSR) Dataset
    """
    def __init__(self, dataset_name="zeroshot", split="train", base_path="project_data/raw/vsr", transform=None):

        # Validations
        self.base_path = Path(base_path or Path(__file__).resolve().parents[1] / "project_data" / "raw" / "vsr") #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert split in ['train', 'dev', 'test'], f"Unsupported split: '{split}'. Must be one of ['train', 'dev', 'test']."
        assert dataset_name in ['zeroshot', 'random'], f"Unsupported vsr name: '{dataset_name}'. Must be one of ['zeroshot', 'random']."
        assert transform is not None, "Transform cannot be None. Please provide a valid transform." 

        
        # Img transformation
        self.transform = transform

        # Get train/dev/test
        self.dataset_name = dataset_name #[zeroshot, random]
        self.base_path = Path(base_path) / self.dataset_name
        self.split = split

        # Load dataset
        data_path = self.base_path / dataset_name / f"{split}.jsonl"
        self.dataset = self._load_jsonl(data_path)

    def _load_jsonl(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try: #try requesting image link
            response = requests.get(item["image_link"], timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load image {item['image_link']}: {e}")
            return None

        return {
            "image": self.transform(image),
            "text": item["caption"],
            "label": item["label"],
            "relation": item["relation"],
        }

    @staticmethod
    def compute_accuracy(preds, labels):
        return (preds.argmax(dim=1) == labels).float().mean() #count coincidences

def get_vsr_loader(dataset_name="zeroshot", split="train", batch_size=8, shuffle=False, transform=None):
    dataset = VSRDataset(dataset_name=dataset_name, split=split, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



class VSRDataModule(pl.LightningDataModule):
    """
    Visual Spatial Relations (VSR) Data Module
    """

    def __init__(self, dataset_name="zeroshot", batch_size=8, transform=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        """
        Called once at the beginning of training, to prepare datasets.
        """
        self.train_dataset = VSRDataset(split="train", dataset_name=self.dataset_name, transform=self.transform)
        self.val_dataset = VSRDataset(split="dev", dataset_name=self.dataset_name, transform=self.transform)
        self.test_dataset = VSRDataset(split="test", dataset_name=self.dataset_name, transform=self.transform)

    def train_dataloader(self):
        return get_vsr_loader(split="train", dataset_name=self.dataset_name,
                              batch_size=self.batch_size, shuffle=True, transform=self.transform)

    def val_dataloader(self):
        return get_vsr_loader(split="dev", dataset_name=self.dataset_name,
                              batch_size=self.batch_size, shuffle=False, transform=self.transform)

    def test_dataloader(self):
        return get_vsr_loader(split="test", dataset_name=self.dataset_name,
                              batch_size=self.batch_size, shuffle=False, transform=self.transform)
