# project_datasets/whatsup_dataset.py
from pathlib import Path
import json
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import torch


# -----------------------------
# DATASETS
# -----------------------------
class WhatsUpDataset(Dataset):
    """
    What's Up Dataset
    """
    def __init__(self, dataset_name="images", data_path="data"):

        # Validations
        self.base_path = Path(data_path) / "raw" / "whatsup" #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert dataset_name in ['images', 'clevr'], f"Unsupported subset: '{dataset_name}'. Must be one of ['images', 'clevr']."
        
        # Get train/dev/test
        self.dataset_name = dataset_name

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
            "image": self._load_image(item["image_path"]),
            "caption_options": item["caption_options"],
            "correct_option": item["caption_options"][0], # The first option is the correct one
        }

    @staticmethod
    def compute_accuracy(logits, labels, score): 
        # TODO acc depending on score
        if score == "pair-wise":
            0
        elif score == "set-wise":
            0
        else: 
            return (logits.argmax(dim=1) == labels).float().mean() 
    


class COCOSpatialDataset(Dataset):
    """
    COCO-spatial Dataset
    """
    def __init__(self, dataset_name="one", data_path="data", image_path="data"):

        # Validations
        self.base_path = Path(data_path) / "raw" / "COCO_spatial" #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert dataset_name in ['one', 'two'], f"Unsupported subset: '{dataset_name}'. Must be one of ['one', 'two']."
        
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
    
class GQASpatialDataset(Dataset):
    """
    GQA-spatial Dataset
    """
    def __init__(self, dataset_name="one", data_path="data", image_path="data"):

        # Validations
        self.base_path = Path(data_path) / "raw" / "GQA_spatial" #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert dataset_name in ['one', 'two'], f"Unsupported subset: '{dataset_name}'. Must be one of ['one', 'two']."
        
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

# -----------------------------
# DATAMODULE
# -----------------------------
class WhatsUpDataModule(pl.LightningDataModule):
    """
    What's Up Data Module
    """
    def __init__(self, args, config): 
        super().__init__()

        self.root = args.root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_name = args.variant 
        self.image_path = args.image_path
        self.dataset = args.dataset
        self.score = args.score

        # Model config
        self.transform = config["transform"]
        self.tokenizer = config["tokenizer"]
        self.processor = config["processor"]
        self.params = config.get("params", {})

        # Prepare data depending on model
        if args.model == "clip":
            self.collate_fn = self.clip_collate

        if args.model in ["siglip", "siglip2"]:
            self.collate_fn = self.siglip_collate

    def setup(self, stage=None):
        """
        Called once at the beginning of training, to prepare datasets.
        """
        if self.dataset == "whatsup":
            self.dataset = WhatsUpDataset(
                score = self.score,
                data_path=self.root,
                dataset_name=self.dataset_name 
            )

        elif self.dataset == "cocospatial":
            self.dataset = COCOSpatialDataset(
                data_path=self.root,
                image_path=self.image_path,
                dataset_name=self.dataset_name
            )

        elif self.dataset == "gqaspatial":
            self.dataset = GQASpatialDataset(
                data_path=self.root,
                image_path=self.image_path,
                dataset_name=self.dataset_name
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

            # Choose correct index
            correct_idx = options.index(correct_caption)
            labels.append(correct_idx)

            # Procesamos todo el texto junto
            inputs = self.processor(
                text=options,
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
    
    def siglip_collate(self, batch):
        labels = []          
        all_inputs = []

        for item in batch:
            options = item["caption_options"]         
            correct_caption = item["correct_option"]  
            img = item["image"]

            # Choose correct index
            correct_idx = options.index(correct_caption)
            labels.append(correct_idx)

            # 1. Transform Image (CLIP processor)
            image_inputs = self.transform(
                images=img,
                return_tensors="pt"
            )
            # 2. Tokenize Text
            text_inputs = self.tokenizer(
                text=options,
                return_tensors="pt",
                **self.params
            )
            # 3. Combine results
            inputs = {**image_inputs, **text_inputs}
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
    