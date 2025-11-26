# project_datasets/whatsup_dataset.py
from pathlib import Path
import json
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
import torch


# -----------------------------
# DATASETS
# -----------------------------
class WhatsUpDataset(Dataset):
    """
    What's Up Dataset
    """
    def __init__(self, dataset_name="images", data_path="data", transform=None, processor=None):


        # Validations
        self.base_path = Path(data_path) / "raw" / "whatsup" #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert dataset_name in ['images', 'clevr'], f"Unsupported subset: '{dataset_name}'. Must be one of ['images', 'clevr']."
        
        # Img transformation
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #  Normalize
        ])

        # Input processor
        self.processor = processor

        # Get train/dev/test
        self.dataset_name = dataset_name
        #self.subset = "A" if self.dataset_name == "images" else "B"

        # Load dataset
        self.data_path = self.base_path / f"controlled_{dataset_name}_dataset.json"
        self.image_path = self.base_path / f"controlled_{dataset_name}"
        self.dataset = self._load_json()

    def _load_json(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
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
            "image": self._load_image(item[0]),
            "caption_options": [str(item[1]), str(item[2])],
            "correct_option": str(item[1]), # The first option is the correct one
        }

    @staticmethod
    def compute_accuracy(preds, labels):
        return (preds.argmax(dim=1) == labels).float().mean() #count coincidences



class COCOSpatialDataset(Dataset):
    """
    COCO-spatial Dataset
    """
    def __init__(self, dataset_name="one", data_path="data", image_path="data", transform=None, processor=None):

        # Validations
        self.base_path = Path(data_path) / "raw" / "COCO_spatial" #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert dataset_name in ['one', 'two'], f"Unsupported subset: '{dataset_name}'. Must be one of ['one', 'two']."
        
        # Img transformation
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #  Normalize
        ])

        # Input processor
        self.processor = processor

        # Get train/dev/test
        self.dataset_name = dataset_name

        # Load dataset
        self.data_path = self.base_path / f"coco_qa_{dataset_name}_obj.json"
        self.image_path = image_path
        self.dataset = self._load_json()

    def _load_json(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_image(self, image):
        img_path = Path(self.image_path) / f"{str(image).zfill(12)}.jpg"
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "image": self._load_image(item[0]),
            "caption_options": [str(item[1]), str(item[2])],
            "correct_option": str(item[1]), # The first option is the correct one
        }

    @staticmethod
    def compute_accuracy(preds, labels):
        return (preds.argmax(dim=1) == labels).float().mean() #count coincidences
    
class GQASpatialDataset(Dataset):
    """
    GQA-spatial Dataset
    """
    def __init__(self, dataset_name="one", data_path="data", image_path="data", transform=None, processor=None):

        # Validations
        self.base_path = Path(data_path) / "raw" / "GQA_spatial" #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert dataset_name in ['one', 'two'], f"Unsupported subset: '{dataset_name}'. Must be one of ['one', 'two']."
        
        # Img transformation
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #  Normalize
        ])

        # Input processor
        self.processor = processor

        # Get train/dev/test
        self.dataset_name = dataset_name

        # Load dataset
        self.data_path = self.base_path / f"vg_qa_{dataset_name}_obj.json"
        self.image_path = image_path
        self.dataset = self._load_json()

    def _load_json(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_image(self, image):
        img_path = Path(self.image_path) / f"{image}.jpg"
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "image": self._load_image(item[0]),
            "caption_options": [str(item[1]), str(item[2])],
            "correct_option": str(item[1]), # The first option is the correct one
        }

    @staticmethod
    def compute_accuracy(preds, labels):
        return (preds.argmax(dim=1) == labels).float().mean() #count coincidences



# -----------------------------
# DATAMODULE
# -----------------------------
class WhatsUpDataModule(pl.LightningDataModule):
    """
    What's Up Data Module
    """
    def __init__(self, args, transform=None, processor=None): 
        super().__init__()

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_name = args.variant 
        self.root = args.root
        self.image_path = args.image_path
        self.dataset = args.dataset

        self.transform = transform
        self.processor = processor

        # Prepare data depending on model
        if args.model == "clip":
            self.collate_fn = self.clip_collate
        elif args.model == "siglip":
            self.collate_fn = self.siglip_collate

    def setup(self, stage=None):
        """
        Called once at the beginning of training, to prepare datasets.
        """
        if self.dataset == "whatsup":
            self.dataset = WhatsUpDataset(
                data_path=self.root,
                dataset_name=self.dataset_name,
                transform=self.transform,
                processor=self.processor
            )

        elif self.dataset == "cocospatial":
            self.dataset = COCOSpatialDataset(
                data_path=self.root,
                image_path=self.image_path,
                dataset_name=self.dataset_name,
                transform=self.transform,
                processor=self.processor
            )

        elif self.dataset == "gqaspatial":
            self.dataset = GQASpatialDataset(
                data_path=self.root,
                image_path=self.image_path,
                dataset_name=self.dataset_name,
                transform=self.transform,
                processor=self.processor
            )
        else: 
            raise NotImplementedError
    
    def clip_collate(self, batch):
        labels = []          
        all_inputs = []

        for item in batch:
            options = item["caption_options"]         
            correct_caption = item["correct_option"]  
            img = item["image"]

            # índice correcto entre de las 4
            correct_idx = options.index(correct_caption)
            labels.append(correct_idx)

            # Procesamos todo el texto junto
            inputs = self.processor(
                text=options,
                images=img,
                return_tensors="pt",
                padding=True
            )
            all_inputs.append(inputs)

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input": all_inputs,
            "label": labels,
        }
    
    def siglip_collate(self, batch):
        labels = []          
        all_inputs = []

        for item in batch:
            options = item["caption_options"]         
            correct_caption = item["correct_option"]  
            img = item["image"]

            # índice correcto entre de las 4
            correct_idx = options.index(correct_caption)
            labels.append(correct_idx)

            # Procesamos todo el texto junto
            inputs = self.processor(
                text=options,
                images=img,
                padding="max_length",
                return_tensors="pt",
            )
            all_inputs.append(inputs)

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input": all_inputs,
            "label": labels,
        }

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    