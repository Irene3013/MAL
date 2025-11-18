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
import random
import torch


negate = {
    # Adjacency
    "adjacent to": "nonadjacent to", 
    "alongside": "away from", 
    "at the side of": "away from", 
    "at the right side of": "at the left side of", 
    "at the left side of": "at the right side of",
    "attached to": "disconnect from", 
    "at the back of": "at the front of", 
    "ahead of": "not ahead of", 
    "against": "away from", 
    "at the edge of": "far from the edge of", 
    # Directional
    "off": "on", 
    "past": "before", 
    "toward": "away from", 
    "down": "up", 
    "away from": "not away from", 
    "along": "not along", 
    "around": "not around", 
    "into": "not into", 
    "across": "not accross",
    "across from": "not across from", 
    "down from": "up from", 
    # Orientation
    "facing": "facing away from", 
    "facing away from": "facing", 
    "parallel to": "perpendicular to", 
    "perpendicular to": "parallel to", 
    # Proximity
    "by": "far away from", 
    "close to": "far from", 
    "near": "far from", 
    "far from": "close to", 
    "far away from": "by", 
    # Topological
    "connected to": "detached from", 
    "detached from": "connected to", 
    "has as a part": "does not have a part", 
    "part of": "not part of", 
    "contains": "does not contain", 
    "within": "outside of", 
    "at": "not at", 
    "on": "not on", 
    "in": "not in",
    "with": "not with", 
    "surrounding": "not surrounding", 
    "among": "not among", 
    "consists of": "does not consists of", 
    "out of": "not out of", 
    "between": "not between", 
    "inside": "outside", 
    "outside": "inside", 
    "touching": "not touching",
    # Unallocated
    "beyond": "inside",
    "next to": "far from", 
    "opposite to": "not opposite to", 
    "enclosed by": "not enclosed by", 
    # missing
    "above": "below",
    "below": "above",
    "behind": "in front of",
    "on top of": "not on top of",
    "under": "over",
    "over": "under",
    "left of": "right of",
    "right of": "left of",
    "in front of": "behind",
    "beneath": "not beneath",
    "beside": "not beside",
    "in the middle of": "not in the middle of",
    "congruent": "incongruent",
}

def invert_relation(caption, relation, inverse_relations):
    """
    Reemplaza la relación en el caption por su opuesta según el diccionario.
    """
    if relation not in inverse_relations:
        raise ValueError(f"There is not a negated relation defined for '{relation}'")
    
    inverse = inverse_relations[relation]
    
    # Reemplazar solo la primera ocurrencia de la relación
    new_caption = caption.replace(relation, inverse, 1)
    
    return new_caption


# -----------------------------
# DATASET
# -----------------------------
class VSRDataset(Dataset):
    """
    Visual Spatial Relations (VSR) Dataset
    """
    def __init__(self, dataset_name="zeroshot", split="train", data_path="data", transform=None, processor=None):

        # Validations
        self.base_path = Path(data_path) / "raw" / "vsr" #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert split in ['train', 'val', 'test'], f"Unsupported split: '{split}'. Must be one of ['train', 'val', 'test']."
        assert dataset_name in ['zeroshot', 'random'], f"Unsupported vsr name: '{dataset_name}'. Must be one of ['zeroshot', 'random']."
        #assert transform is not None, "Transform cannot be None. Please provide a valid transform." 
        
        # Img transformation
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Convierte la imagen PIL a un tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza la imagen
        ])

        self.processor = processor

        # Get train/val/test
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

    def clip_collate(batch):
        images = []
        texts = []
        labels = []

        for item in batch:
            images.append(item["image"])

            # Si texto es lista (multi-caption), extender
            if isinstance(item["text"], list):
                texts.extend(item["text"])
            else:
                texts.append(item["text"])

            labels.append(item["label"])

        images = torch.stack(images)
        return {
            "images": images,
            "texts": texts,
            "labels": torch.stack(labels),
        }


    def __getitem__(self, idx):
        item = self.dataset[idx]
        try: #try requesting image link
            #response = requests.get(item["image_link"], timeout=5)
            #image = Image.open(BytesIO(response.content)).convert("RGB")
            image = Image.open(requests.get(item["image_link"], stream=True).raw)
        except Exception as e:
            print(f"[WARN] Failed to load image {item['image_link']}: {e}")
            new_idx = random.randint(0, len(self.dataset)-1)
            return self.__getitem__(new_idx)

        negated = invert_relation(item["caption"], item["relation"], negate)
        label = torch.tensor([item["label"], 1 - item["label"]])

        if self.model == "clip":

            input = self.processor(text=[item["caption"], negated], images=image, return_tensors="pt", padding=True)
            return {
            "input": input,
            "label": label,                     # dim=2: 1-TRUE / 0-FALSE
        }
        
        return {
            "image": self.transform(image),
            "text": [item["caption"], negated], # Both captions
            "label": label,                     # dim=2: 1-TRUE / 0-FALSE
        }
        

    @staticmethod
    def compute_accuracy(logits, labels, mode="multicaption"):
        if mode=="singlecaption":
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            return (preds == labels).float().mean()
        else:
            return (logits.argmax(dim=1) == labels).float().mean()

def get_vsr_loader(data_path="data", dataset_name="zeroshot", split="train", batch_size=8,
                   shuffle=False, transform=None, num_workers=0, processor=None):
    dataset = VSRDataset(dataset_name=dataset_name, split=split, data_path=data_path, transform=transform, processor=processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)




# -----------------------------
# DATAMODULE
# -----------------------------
class VSRDataModule(pl.LightningDataModule):
    """
    Visual Spatial Relations (VSR) Data Module
    """
    def __init__(self, args, transform=None, processor=None):
        super().__init__()

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_name = args.variant # [zeroshot / random]
        self.root = args.root
        self.negated = args.clip_mode == "multicaption"

        self.transform = transform
        self.processor = processor

    def setup(self, stage=None):
        """
        Called once at the beginning of training, to prepare datasets.
        """
        self.train_dataset = VSRDataset(split="train", data_path=self.root, dataset_name=self.dataset_name, transform=self.transform)
        self.val_dataset = VSRDataset(split="val", data_path=self.root, dataset_name=self.dataset_name, transform=self.transform)
        self.test_dataset = VSRDataset(split="test", data_path=self.root, dataset_name=self.dataset_name, transform=self.transform)

    def train_dataloader(self):
        params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'negated': self.negated,
            'num_workers': self.num_workers,
            'dataset_name': self.dataset_name,
            'transform': self.transform,
            'data_path': self.root,
            'processor': self.processor
        }
        return get_vsr_loader(split="train", **params)

    def val_dataloader(self):
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'negated': self.negated,
            'num_workers': self.num_workers,
            'dataset_name': self.dataset_name,
            'transform': self.transform,
            'data_path': self.root,
            'processor': self.processor
        }
        return get_vsr_loader(split="val", **params)

    def test_dataloader(self):
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'negated': self.negated,
            'num_workers': self.num_workers,
            'dataset_name': self.dataset_name,
            'transform': self.transform,
            'data_path': self.root,
            'processor': self.processor
        }
        return get_vsr_loader(split="test", **params)
