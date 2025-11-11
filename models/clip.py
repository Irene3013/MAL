#models/clip.py
import torch
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from project_datasets.vsr_dataset import get_vsr_loader
import pytorch_lightning as pl
#from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.callbacks import ModelCheckpoint
from project_datasets.vsr_dataset import get_vsr_loader, VSRDataset


class ClipModel(pl.lightningmodel):
    """
    Wrapper Lightning Module for CLIP model fine-tuning or zero-shot evaluation.
    """

    def __init__(self, model_name="ViT-B/32", dataset="vsr", batch_size=8, lr=1e-5, mode="zeroshot"):
        super().__init__()

        self.save_hyperparameters()

        # --- Validations ---
        available_models = [
            'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 
            'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        assert model_name in available_models, f"Unsupported clip version: '{model_name}'. Must be one of ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']."

        # --- Params ---
        self.model_name = model_name # ViT-B/32 
        self.dataset = dataset       # vsr, whatsup biscor
        self.batch_size = batch_size
        self.lr = lr
        self.mode = mode  # 'zeroshot' or 'finetune'
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # --- Load CLIP ---
        self.model, self.preprocess = clip.load(self.model_name, device="cpu")#TODO change to GPU
        self.transform = self.preprocess

        # --- Load CLIP Tokenizer ---
        self.tokenizer = clip.tokenize()

        # --- Get dataset class ---
        if self.dataset == "vsr":
            self.dataset_class = VSRDataset
        else: 
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        # --- Custom transform (matches CLIP preprocessing) ---
        #self.transform = Compose(
        #    Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
        #    CenterCrop(size=(224, 224))
        #    <function _transform.<locals>.<lambda> at 0x7f3a24ffb440>
        #    ToTensor()
        #    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
        #              std=(0.26862954, 0.26130258, 0.27577711))
        #)        


    # -----------------------------
    # COMPUTE LOSS
    # -----------------------------
    def compute_loss(self, logits, labels):
        """
        Compute contrastive loss for CLIP.
        """
        return self.loss_fn(logits, labels)

    #def compute_accuracy(self, batch)
    #    #TODO acuracy


    # -----------------------------
    # FORWARD PASS
    # -----------------------------
    def forward(self, images, texts):
        """
        Forward pass through CLIP.
        """
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)

        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        return logits


    # -----------------------------
    # STEP (train/val/test)
    # -----------------------------
    def step(self, batch):
        images, texts, labels = batch["image"], batch["text"], batch["label"]
        
        # Tokenize text
        tokenized_texts = clip.tokenize(texts, truncate=True).to(self.device)

        # Forward
        logits = self.forward(images.to(self.device), tokenized_texts)

        # Loss
        loss = self.compute_loss(logits, labels.to(self.device))
        
        # Accuracy -- depends on the dataset
        accuracy = self.dataset_class.compute_accuracy(logits, labels)

        #Logging
        #self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
        #self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)

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
        return optimizer
