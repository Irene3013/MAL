# data/processed/relations_dataset.py

from pathlib import Path
import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
from utils.data_helpers import biscor_dual_encoder_collate
# from utils.model_helpers import create_qwen_messages_biscor
# from qwen_vl_utils import process_vision_info


# -----------------------------
# DATASET
# -----------------------------
class RELDataset(Dataset):
    """
    BISCOR Dataset
    """
    def __init__(self, version="v1", split="train", data_path="data", model=None, config=None):

        # Validations
        self.data_path = Path(data_path) #relative path
        assert self.data_path.exists(), f"Root directory does not exist."   
        assert split in ['train', 'val', 'test'], f"Unsupported split: '{split}'. Must be one of ['train', 'val', 'test']."
        
        # Data / Images path
        self.model = model 
        self.version = version
        mapping = {
            "v4": "v3",
            "v6": "v5",
            "v9": "v8",
        }
        image_version = mapping.get(self.version, self.version)
        self.split = split 

        # ---- IMAGE PATH ----
        img_folder = "test_images" if self.split == "test" else "train_images"
        self.image_path = Path(data_path) / image_version / img_folder

        # ---- CSV PATH ----
        if self.version == 'v6':
            if self.split == 'test':
                csv_name = "v5_test_paraphrase.csv"
            else:
                csv_name = f"v5_{self.split}.csv"
            self.data_path = Path(data_path) / "v5" / csv_name
        else:
            self.data_path = Path(data_path) / self.version / f"{self.version}_{self.split}.csv"
        
        self.dataset = self._load_csv()

        # Input processing
        if config is not None:
            self.transform = config["transform"]
            self.tokenizer = config["tokenizer"]
            self.processor = config["processor"]
            self.params = config.get("params", {})

    def _load_csv(self):
        with open(self.data_path, newline="\n", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            return list(reader)
    
    def _load_image(self, image):
        img_path = self.image_path / Path(image)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.model in ["clip", "siglip", "siglip2", "pecore"]: # Dual encoder
            return self._dual_encoder_item(item)
        else:
            raise NotImplementedError()
        
    
    # --- GET ITEM METHODS ---
    def _dual_encoder_item(self, item):
        """
        Prepare item for Dual Encoder models.
            Val/Test: 2 image - 2 captions
            Train:    2 image - 2 caption (contrastive loss)
        """    
        pos_capt,neg_capt,relation,shape1,color1,shape2,color2,image = item

        pos_img = f'pos_{image}'
        neg_img = f'neg_{image}'

        # **A. Val/Test (return caption-pairs to Collate):**
        #if self.split != "train":
        return {
            "caption_pos": pos_capt,
            "caption_neg": neg_capt,
            "image_pos": self._load_image(pos_img),
            "image_neg": self._load_image(neg_img),
        }
    


# -----------------------------
# DATAMODULE
# -----------------------------
class RELDataModule(pl.LightningDataModule):
    """
    Visual Spatial Relations (BISCOR) Data Module
    """
    def __init__(self, args, config):
        super().__init__()

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_name = args.variant
        self.root = args.root
        self.model = args.model
        self.config = config
        self.version = args.variant

        # Setup dataloader
        self.setup()
        
    def setup(self, stage=None):
        # Define collate function (for evaluation)
        if self.model in ["clip", "siglip", "siglip2", "pecore"]: # Dual Encoders
            self.collate_fn_eval = lambda batch: biscor_dual_encoder_collate(
                batch, self.config, self.model # Pasar args y model_name
            )
        else: # qwen / vqascore
            self.collate_fn_eval = None

        # Setup train/val/test datasets
        self.train_dataset = RELDataset(
            version = self.version,
            split="train",
            data_path=self.root,
            model=self.model,
            config=self.config 
        )

        self.val_dataset = RELDataset(
            version = self.version,
            split="val",
            data_path=self.root,
            model=self.model,
            config=self.config 
        )

        self.test_dataset = RELDataset(
            version = self.version,
            split="test",
            data_path=self.root,
            model=self.model,
            config=self.config 
        )

    def length(self):
        return self.train_dataset.__len__()

    # DATALOADERS #
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_eval
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_eval
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_eval
        )