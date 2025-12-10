#models/dual_encoder.py
import torch
import pytorch_lightning as pl
from project_datasets.vsr_dataset import VSRDataset
from project_datasets.whatsup_dataset import WhatsUpDataset
from torchvision import transforms
from torchvision.transforms import Resize, CenterCrop
import transformers
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as coreTransforms

class DualEncoder(pl.LightningModule):
    """
    Wrapper Lightning Module for CLIP model fine-tuning or zero-shot evaluation.
    """
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()

         # --- Params ---
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps
        self.lr = args.lr
        self.scheduler_off = args.scheduler_off
        self.batch_size = args.batch_size
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.score = args.score

        print(f"args.gpus: {args.gpus}")
        self.device_name = "cpu" if args.gpus == 0 else "cuda"

        # CLIP image processor
        self.preprocess = transforms.Compose([
            Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(224),
        ])

        # --- Load Model ---
        if args.model == "clip":
            # https://huggingface.co/openai/clip-vit-base-patch32
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.config = {
                "processor": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
                "transform": None,
                "tokenizer": None,
                "params": {"padding": "max_length", "max_length": 64}

                #"params": {"padding": True, "truncation": True}
            }

        elif args.model == "siglip":
            # https://huggingface.co/google/siglip-base-patch16-224
            self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
            self.config = {
                "processor": AutoProcessor.from_pretrained("google/siglip-base-patch16-224"),
                "transform": self.preprocess, # crop images for comparable results
                "tokenizer": None,
                "params": {"padding": "max_length", "max_length": 64}
            }

        elif args.model == "siglip2":
            # https://huggingface.co/google/siglip2-base-patch32-256
            self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
            self.config = {
                "processor": AutoProcessor.from_pretrained("google/siglip2-base-patch16-224"),
                "transform": self.preprocess, # crop images for comparable results
                "tokenizer": None,
                "params": {"padding": "max_length", "max_length": 64}
            }

        elif args.model == "pecore":
            # https://huggingface.co/facebook/PE-Core-B16-224
            self.model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)
            self.model.to(self.device)
            self.config = {
                "processor": coreTransforms.get_image_transform(self.model.image_size),
                "transform": self.preprocess, # crop images for comparable results
                "tokenizer": coreTransforms.get_text_tokenizer(self.model.context_length),
                "params": None
            }

        else:
            raise NotImplementedError
        
        self.model_name = args.model

        # --- Accuracy depending on dataset ---
        if args.dataset == "whatsup":
            # precision / pair-wise / set-wise
            self.compute_accuracy = WhatsUpDataset.compute_accuracy
        else:
            # precision
            self.compute_accuracy = VSRDataset.compute_accuracy

    # -----------------------------
    # STEP (train/val/test)
    # -----------------------------
    def step(self, batch, split):
        if split == "train":
            return self.train_step(batch, split)
        else:
            return self.eval_step(batch, split)

    def train_step(self, batch, split):
        images = batch["image"].to(self.device)
        texts  = batch["text"].to(self.device)

        outputs = self.model(texts, images) 
        logits_per_image, logits_per_text  = outputs.logits_per_image, outputs.logits_per_text

        batch_size = images.size(0)
        ground_truth = torch.arange(batch_size, device=self.device, dtype=torch.long) # Every caption is true

        loss = 0.5 * (
            self.cross_entropy(logits_per_image, ground_truth) +
            self.cross_entropy(logits_per_text,  ground_truth)
        )

        # Logging
        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
        return loss


    def eval_step(self, batch, split):
        labels = batch["label"].to(self.device)
        inputs_list = batch["input"]

        # Forward pass each input
        logits_list = []
        for inputs in inputs_list:

            if self.model_name == "pecore":
                image = inputs['image'].to(self.device)
                captions = inputs['captions'].to(self.device)
                image_features, text_features, logit_scale = self.model(image, captions)
                I2T_logits = logit_scale * image_features @ text_features.T

            else:
                inputs = inputs.to(self.device)
                outputs = self.model(**inputs)
                I2T_logits = outputs.logits_per_image

            logits_list.append(I2T_logits)

        logits = torch.cat(logits_list, dim=0)
        acc = self.compute_accuracy(logits, labels, self.score)

        # Logging
        self.log(f'{split}_accuracy', acc, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
        return acc


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