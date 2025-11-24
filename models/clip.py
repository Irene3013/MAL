#models/clip.py
import torch
#import clip
import pytorch_lightning as pl
from project_datasets.vsr_dataset import VSRDataset
import transformers
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
from PIL import Image

class ClipModel(pl.LightningModule):
    """
    Wrapper Lightning Module for CLIP model fine-tuning or zero-shot evaluation.
    """
    def __init__(self, args): 
        super().__init__()

        self.save_hyperparameters()

         # --- Params ---
        self.model_name = args.target_model # Version of the CLIP model
        self.dataset = args.dataset         # [vsr, whatsup, biscor]
        self.batch_size = args.batch_size
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        print(f"args.gpus: {args.gpus}")
        self.device_name = "cpu" if args.gpus == 0 else "cuda"


        # --- Validations ---
        available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64','ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        assert self.model_name in available_models, f"Unsupported clip model: '{self.model_name}'. Must be one of [{', '.join(available_models)}]."

        # --- Load CLIP ---
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # --- Get dataset class ---
        # if self.dataset == "vsr":
        #     self.dataset_class = VSRDataset
        # else: 
        #     raise ValueError(f"Unsupported dataset: {self.dataset}")

        # --- Define other hyperparameters ---
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps
        self.lr = args.lr
        self.scheduler_off = args.scheduler_off

    # -----------------------------
    # COMPUTE LOSS
    # -----------------------------
    def compute_loss(self, logits, labels):
        """
        Compute contrastive loss for CLIP.
        """
        return self.loss_fn(logits, labels)

    # -----------------------------
    # STEP (train/val/test)
    # -----------------------------
    def step(self, batch, split):
        labels = batch["label"].to(self.device)
        inputs_list = batch["input"]
        logits_list = []
        
        # Forward pass each input
        for inputs in inputs_list:     
            inputs = inputs.to(self.device)
            outputs = self.model(**inputs)
            logits_list.append(outputs.logits_per_image)

        # Loss and accuracy
        logits = torch.cat(logits_list, dim=0)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        pred = logits.argmax(dim=1)
        accuracy = (pred == labels).float().mean()

        # Logging
        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
        self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
        
        return loss

    # -----------------------------
    # LIGHTNING STEP METHODS
    # -----------------------------
    def training_step(self, batch, batch_idx):
        return self.step(batch, split="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, split="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.scheduler_off:
            return [optimizer]
        else:
            scheduler = {
                "scheduler": transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps),
                "interval": "step"
            }
            return [optimizer], [scheduler]