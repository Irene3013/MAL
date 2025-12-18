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
from utils.data_helpers import invert_relation, vsr_dual_encoder_collate
from utils.model_helpers import create_qwen_message, create_MC_qwen_message
from qwen_vl_utils import process_vision_info

# -----------------------------
# DATASET
# -----------------------------
class VSRDataset(Dataset):
    """
    Visual Spatial Relations (VSR) Dataset
    """
    def __init__(self, dataset_name="zeroshot", split="train", data_path="data", model=None, config=None):

        # Validations
        self.base_path = Path(data_path) / "raw" / "vsr" #relative path
        assert self.base_path.exists(), f"Root directory '{self.base_path}' does not exist."   
        assert split in ['train', 'val', 'test'], f"Unsupported split: '{split}'. Must be one of ['train', 'val', 'test']."
        assert dataset_name in ['zeroshot', 'random'], f"Unsupported vsr name: '{dataset_name}'. Must be one of ['zeroshot', 'random']."
        
        # Data / Images path
        self.dataset_name = dataset_name # [zeroshot, random]
        self.split = split
        self.model = model

        self.image_path = Path(self.base_path) / "images"
        self.data_path = Path(self.base_path) / self.dataset_name  / f"{split}.jsonl"
        self.dataset = self._load_jsonl()

        # Input processing
        self.transform = config["transform"]
        self.tokenizer = config["tokenizer"]
        self.processor = config["processor"]
        self.params = config.get("params", {})
        self.score = "mc"

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

        if self.model in ["clip", "siglip", "siglip2", "pecore"]:
            return self._dual_encoder_item(item)
        
        elif self.model in ["qwen2"]:
            return self._qwen_item(item)
            
        else:
            raise NotImplementedError()


    # --- ITEM METHODS ---
    def _dual_encoder_item(self, item):
        """
        Prepare item for Dual Encoder models.
            Val/Test --> 1 image n captions
            Train --> 1 image 1 caption (contrastive loss)
        """    
        # **A. Val/Test (return caption-pairs to Collate):**
        if self.split != "train":
            return {
                "caption": item["caption"],
                "negated": invert_relation(item["caption"], item["relation"]),
                "image": self._load_image(item["image"]),
                "label": item["label"]
            }

        # **B. Train (direct input preprocessing):**
        img = self._load_image(item["image"])
        text = item["caption"] + (' (True)' if item['label'] == 1 else ' (False)')

        if self.transform is not None: # CLIP transform if specified
            img = self.transform(img)

        if self.tokenizer: # PE-core (image processor + text tokenizer)
            img_tensor = self.processor(img).unsqueeze(0)
            text_tokens = self.tokenizer(text)
            return {
                "image": img_tensor, 
                "text": text_tokens
            }
        else: # CLIP/SigLIP (unified processor)
            inputs = self.processor(
                text=text,
                images=img,
                return_tensors="pt",
                **self.params
            )
            return {
                "image": inputs['pixel_values'].squeeze(0), # delete extra batch dim.
                "text": inputs['input_ids'],
            }


    def _qwen_item(self, item):
        """Prepare item for Qwen2-VL model."""
        img_path = self.image_path / item["image"]
        caption = item["caption"]
        negated = invert_relation(caption, item["relation"])
        

        #if self.score == "mc":
        messages = create_MC_qwen_message(img_path, [caption, negated])
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Expected response to be generated by Qwen (ej: "A" or "B")
        expected_response = "A" if item["label"] == 1 else "B"

        # else:
        #     inputs = []
        #     messages = create_qwen_message(img_path, [caption, negated])
        #     for message in messages: #Process image - message in pairs
        #         text = self.processor.apply_chat_template(
        #             message, tokenize=False, add_generation_prompt=True
        #         )
        #         image_inputs, video_inputs = process_vision_info(message)
        #         input = self.processor(
        #             text=[text],
        #             images=image_inputs,
        #             videos=video_inputs,
        #             padding=True,
        #             return_tensors="pt",
        #         )
        #         inputs.append(input)

        #     # Expected response to be generated by Qwen (ej: "True" or "False" for each)
        #     expected_response = ["True", "False"] if item["label"] == 1 else ["False", "True"]

        return {
             "input": inputs,
             "label": expected_response 
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
        self.model = args.model
        self.config = config

        # Setup dataloader
        self.setup()
        
    def setup(self, stage=None):
        # Define collate function (for evaluation)
        if self.model in ["clip", "siglip", "siglip2", "pecore"]: # Dual Encoders
            self.collate_fn_eval = lambda batch: vsr_dual_encoder_collate(
                batch, self.config, self.model # Pasar args y model_name
            )
        else: 
            self.collate_fn_eval = None

        # Setup train/val/test datasets
        self.train_dataset = VSRDataset(
            split="train",
            data_path=self.root,
            dataset_name=self.dataset_name,
            model=self.model,
            config=self.config 
        )
        self.val_dataset = VSRDataset(
            split="val",
            data_path=self.root,
            dataset_name=self.dataset_name,
            model=self.model,
            config=self.config 
        )
        self.test_dataset = VSRDataset(
            split="test",
            data_path=self.root,
            dataset_name=self.dataset_name,
            model=self.model,
            config=self.config 
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
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