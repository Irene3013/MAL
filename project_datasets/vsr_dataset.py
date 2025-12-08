# project_datasets/vsr_dataset.py
from pathlib import Path
import json
import requests
from io import BytesIO
import os
import torchvision
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
    "through": "not through",
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
    def __init__(self, dataset_name="zeroshot", split="train", data_path="data"):

        # Validations
        self.base_path = Path(data_path) / "raw" / "vsr" #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert split in ['train', 'val', 'test'], f"Unsupported split: '{split}'. Must be one of ['train', 'val', 'test']."
        assert dataset_name in ['zeroshot', 'random'], f"Unsupported vsr name: '{dataset_name}'. Must be one of ['zeroshot', 'random']."
        
        # Data / Images path
        self.dataset_name = dataset_name # [zeroshot, random]
        self.split = split

        self.image_path = Path(self.base_path) / "images"
        self.data_path = Path(self.base_path) / self.dataset_name  / f"{split}.jsonl"
        self.dataset = self._load_jsonl()

    def _load_jsonl(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    
    def _load_image(self, image):
        img_path = self.image_path / image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        return {
            "caption": item["caption"],
            "negated": invert_relation(item["caption"], item["relation"], negate),
            "image": self._load_image(item["image"]),
            "label": item["label"]
        }

    @staticmethod
    def compute_accuracy(logits, labels, score):
        probs = torch.sigmoid(logits)
        return (probs.argmax(dim=1) == labels).float().mean()


# -----------------------------
# DATAMODULE
# -----------------------------
class VSRDataModule(pl.LightningDataModule):
    """
    Visual Spatial Relations (VSR) Data Module
    """
    def __init__(self, args, config):
        super().__init__()

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_name = args.variant
        self.root = args.root

        # Model config
        self.transform = config["transform"]
        self.tokenizer = config["tokenizer"]
        self.processor = config["processor"]
        self.params = config.get("params", {})

        if args.model == "clip":
            self.collate_fn = self.clip_collate

        if args.model in ["siglip", "siglip2"]:
            self.collate_fn = self.siglip_collate

    def setup(self, stage=None):
      self.train_dataset = VSRDataset(
          split="train",
          data_path=self.root,
          dataset_name=self.dataset_name
      )
      self.val_dataset = VSRDataset(
          split="val",
          data_path=self.root,
          dataset_name=self.dataset_name
      )
      self.test_dataset = VSRDataset(
          split="test",
          data_path=self.root,
          dataset_name=self.dataset_name
      )

    def pe_collate(self, batch):
        labels = []          
        all_inputs = []

        for item in batch:
            caption = item["caption"]         
            negation = item["negated"]  
            img = item["image"]

            # Choose correct index
            correct_idx = 0 if item["label"] == 1 else 1
            labels.append(correct_idx)

            # 1. CROP image (like CLIP)
            img_crop = self.transform(img)
            img_crop = img_crop.convert("RGB")

            # 2. Process inputs
            image = self.processor(img_crop).unsqueeze(0)
            text = self.tokenizer([caption, negation])
            inputs = {'pixel_values': image, 'input_ids': text}
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
            caption = item["caption"]         
            negation = item["negated"]  
            img = item["image"]

            # Choose correct index
            correct_idx = 0 if item["label"] == 1 else 1
            labels.append(correct_idx)

            # 1. CROP image (like CLIP)
            img_crop = self.transform(img)
            img_crop = img_crop.convert("RGB")

            # 2. Process inputs
            inputs = self.processor(
                text=[caption, negation],
                images=img_crop,
                return_tensors="pt",
                **self.params
            )
            all_inputs.append(inputs)

        labels = torch.tensor(labels, dtype=torch.long)
        return {
            "input": all_inputs,
            "label": labels,
        }

    def clip_collate(self, batch):
        labels = []          
        all_inputs = []

        for item in batch:
            caption = item["caption"]         
            negation = item["negated"]  
            img = item["image"]

            # Choose correct index
            correct_idx = 0 if item["label"] == 1 else 1
            labels.append(correct_idx)

            # Process inputs
            inputs = self.processor(
                text=[caption, negation],
                images=img,
                return_tensors="pt",
                **self.params
            )
            all_inputs.append(inputs)

        labels = torch.tensor(labels, dtype=torch.long)
        return {
            "input": all_inputs,
            "label": labels,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )