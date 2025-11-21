#models/clip.py
import torch
#import clip
import pytorch_lightning as pl
from project_datasets.vsr_dataset import VSRDataset
import transformers
from torchvision import transforms

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
        self.mode = args.clip_mode 
        self.to_pil = transforms.ToPILImage()
        
        if self.mode == "singlecaption":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
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
        if self.dataset == "vsr":
            self.dataset_class = VSRDataset
        else: 
            raise ValueError(f"Unsupported dataset: {self.dataset}")

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
        - BCE if singlecaption
        - CE if multicaption
        """
        return self.loss_fn(logits, labels)

    # # -----------------------------
    # # FORWARD PASS
    # # -----------------------------
    # def forward(self, images, texts):
    #     """
    #     Forward pass through CLIP: 
    #     - multicaption: 1 image + N caption.
    #     - singlecaption: 1 image + 1 caption.
    #     """
    #     # image_features = self.model.encode_image(images)
    #     # text_features = self.model.encode_text(texts)

    #     # # Normalize for cosine similarity
    #     # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    #     # if self.mode =="multicaption":
    #     #     return 100.0 * image_features @ text_features.T
    #     # else:
    #     #     return (image_features * text_features).sum(dim=1) * 100.0


    # -----------------------------
    # STEP (train/val/test)
    # -----------------------------
    def step(self, batch, split):
        inputs = batch["input"].to(self.device)
        labels = batch["label"].to(self.device)
    

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image   # Shape: (N_images, 2) porque cada imagen tiene 2 captions

        
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        pred = logits_per_image.argmax(dim=1)  # Devuelve el índice del caption con mayor similitud
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
