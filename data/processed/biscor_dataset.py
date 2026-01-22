# data/processed/biscor_dataset.py

from pathlib import Path
import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
from utils.data_helpers import biscor_dual_encoder_collate
# from utils.model_helpers import create_qwen_message, create_MC_qwen_message
# from qwen_vl_utils import process_vision_info


# -----------------------------
# DATASET
# -----------------------------
class BISCORDataset(Dataset):
    """
    BISCOR Dataset
    """
    def __init__(self, split="train", data_path="data", model=None, config=None):

        # Validations
        self.data_path = Path(data_path) #relative path
        assert self.data_path.exists(), f"Root directory does not exist."   
        assert split in ['train', 'val', 'test'], f"Unsupported split: '{split}'. Must be one of ['train', 'val', 'test']."
        
        # Data / Images path
        self.model = model
        self.split = split
        self.image_path = Path(data_path)
        self.data_path = Path(data_path) / "rel" / f"test_relation.csv"
        self.dataset = self._load_csv()

        # Input processing
        self.transform = config["transform"]
        self.tokenizer = config["tokenizer"]
        self.processor = config["processor"]
        self.params = config.get("params", {})

    def _load_csv(self):
        with open(self.data_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
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
        
        # elif self.model == "qwen2":
        #     return self._qwen_item(item)
        
        # elif self.model == "clip-flant5":
        #     return self._vqascore_item(item)
        else:
            raise NotImplementedError()
        
    
    # --- GET ITEM METHODS ---
    def _dual_encoder_item(self, item):
        """
        Prepare item for Dual Encoder models.
            Val/Test: 2 image - 2 captions
            Train:    2 image - 2 caption (contrastive loss)
        """    
        _, pos_capt, neg_capt, pos_img, neg_img  = item

        # **A. Val/Test (return caption-pairs to Collate):**
        if self.split != "train":
            return {
                "caption_pos": pos_capt,
                "caption_neg": neg_capt,
                "image_pos": self._load_image(pos_img),
                "image_neg": self._load_image(neg_img),
            }

    @staticmethod
    def compute_accuracy(outputs, labels, score):
        t2i = outputs.logits_per_image
        i2t = outputs.logits_per_text

        probs_t2i = torch.sigmoid(t2i)
        probs_i2t = torch.sigmoid(i2t)
        acc_t2i = (probs_t2i.argmax(dim=1) == labels).float().mean()
        acc_i2t = (probs_i2t.argmax(dim=1) == labels).float().mean()

        return (acc_t2i == 1) and (acc_i2t == 1) # must guess all correctly
    

    # -----------------------------
# DATAMODULE
# -----------------------------
class BISCORDataModule(pl.LightningDataModule):
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
        # self.train_dataset = BISCORDataset(
        #     split="train",
        #     data_path=self.root,
        #     dataset_name=self.dataset_name,
        #     model=self.model,
        #     config=self.config 
        # )
        # self.val_dataset = BISCORDataset(
        #     split="val",
        #     data_path=self.root,
        #     dataset_name=self.dataset_name,
        #     model=self.model,
        #     config=self.config 
        # )
        self.test_dataset = BISCORDataset(
            split="test",
            data_path=self.root,
            model=self.model,
            config=self.config 
        )

    # def train_dataloader(self):
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=self.num_workers,
    #     )
    
    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         collate_fn=self.collate_fn_eval
    #     )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_eval
        )