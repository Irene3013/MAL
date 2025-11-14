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

# -----------------------------
# DATASET
# -----------------------------
class VSRDataset(Dataset):
    """
    Visual Spatial Relations (VSR) Dataset
    """
    def __init__(self, dataset_name="zeroshot", split="train", data_path="project_data", transform=None):

        # Validations
        self.base_path = Path(data_path) / "raw" / "vsr" #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert split in ['train', 'dev', 'test'], f"Unsupported split: '{split}'. Must be one of ['train', 'dev', 'test']."
        assert dataset_name in ['zeroshot', 'random'], f"Unsupported vsr name: '{dataset_name}'. Must be one of ['zeroshot', 'random']."
        assert transform is not None, "Transform cannot be None. Please provide a valid transform." 

        
        # Img transformation
        self.transform = transform

        # Get train/dev/test
        self.dataset_name = dataset_name #[zeroshot, random]
        self.base_path = Path(self.base_path) / self.dataset_name
        self.split = split

        # Load dataset
        data_path = self.base_path / f"{split}.jsonl"
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
            "label": item["label"], # 1-TRUE / 0-FALSE
        }

    @staticmethod
    def compute_accuracy(preds, labels):
        return (preds.argmax(dim=1) == labels).float().mean() #count coincidences

def get_vsr_loader(data_path="project_data", dataset_name="zeroshot", split="train", batch_size=8,
                   shuffle=False, transform=None, num_workers=0):
    dataset = VSRDataset(dataset_name=dataset_name, split=split, data_path=data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)




# -----------------------------
# DATAMODULE
# -----------------------------
class VSRDataModule(pl.LightningDataModule):
    """
    Visual Spatial Relations (VSR) Data Module
    """
    def __init__(self, args, transform=None):
        super().__init__()

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_name = args.variant # [zeroshot / random]
        self.transform = transform
        self.root = args.root

    def setup(self, stage=None):
        """
        Called once at the beginning of training, to prepare datasets.
        """
        self.train_dataset = VSRDataset(split="train", data_path=self.root, dataset_name=self.dataset_name, transform=self.transform)
        self.val_dataset = VSRDataset(split="dev", data_path=self.root, dataset_name=self.dataset_name, transform=self.transform)
        self.test_dataset = VSRDataset(split="test", data_path=self.root, dataset_name=self.dataset_name, transform=self.transform)

    def train_dataloader(self):
        params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'dataset_name': self.dataset_name,
            'transform': self.transform,
            'data_path': self.root
        }
        return get_vsr_loader(split="train", **params)

    def val_dataloader(self):
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'dataset_name': self.dataset_name,
            'transform': self.transform,
            'data_path': self.root
        }
        return get_vsr_loader(split="dev", **params)

    def test_dataloader(self):
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'dataset_name': self.dataset_name,
            'transform': self.transform,
            'data_path': self.root
        }
        return get_vsr_loader(split="test", **params)
