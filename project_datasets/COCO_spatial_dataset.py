# project_datasets/COCO-spatial_dataset.py
from pathlib import Path
import json
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image


class COCO_SpatialDataset(Dataset):
    """
    What's Up Dataset
    """
    def __init__(self, dataset_name="one", base_path="project_data/raw/COCO_spatial", transform=None):

        # Validations
        self.base_path = Path(base_path or Path(__file__).resolve().parents[1] / "project_data" / "raw" / "whatsup") #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert dataset_name in ['one', 'two'], f"Unsupported subset: '{dataset_name}'. Must be one of ['one', 'two']."
        assert transform is not None, "Transform cannot be None. Please provide a valid transform." 

        
        # Img transformation
        self.transform = transform

        # Get train/dev/test
        self.dataset_name = dataset_name
        self.base_path = Path(base_path) / self.dataset_name
        self.subset = "A" if self.dataset_name == "images" else "B"

        # Load dataset
        self.data_path = self.base_path / f"controlled_{dataset_name}_dataset.jsonl"
        self.image_path = self.base_path / f"controlled_{dataset_name}"
        self.dataset = self._load_jsonl()

    def _load_jsonl(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    
    def _load_image(self, orig_path):
        img_path = self.image_path / orig_path.split("/")[-1]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "image": self.transform(self._load_image(item["image_path"])),
            "caption_options": item["caption_options"],
            "correct_option": item["caption_options"][0], # The first option is the correct one
            #"relation": item["relation"],
        }

    @staticmethod
    def compute_accuracy(preds, labels):
        return (preds.argmax(dim=1) == labels).float().mean() #count coincidences

def get_whatsup_loader(dataset_name="images", batch_size=8, shuffle=False, transform=None):
    dataset = COCO_SpatialDataset(dataset_name=dataset_name, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# -----------------------------
# DATAMODULE
# -----------------------------
class COCO_SpatialDataModule(pl.LightningDataModule):
    """
    Visual Spatial Relations (VSR) Data Module
    """

    def __init__(self, dataset_name="images", batch_size=8, transform=None): #TODO args
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        """
        Called once at the beginning of training, to prepare datasets.
        """
        self.train_dataset = COCO_SpatialDataset(dataset_name=self.dataset_name, transform=self.transform)
        self.val_dataset = COCO_SpatialDataset(dataset_name=self.dataset_name, transform=self.transform)
        self.test_dataset = COCO_SpatialDataset(dataset_name=self.dataset_name, transform=self.transform)

    def train_dataloader(self):
        return get_whatsup_loader(dataset_name=self.dataset_name,
                              batch_size=self.batch_size, shuffle=True, transform=self.transform)

    def val_dataloader(self):
        return get_whatsup_loader(dataset_name=self.dataset_name,
                              batch_size=self.batch_size, shuffle=False, transform=self.transform)

    def test_dataloader(self):
        return get_whatsup_loader(dataset_name=self.dataset_name,
                              batch_size=self.batch_size, shuffle=False, transform=self.transform)