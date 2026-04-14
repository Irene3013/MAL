#models/dual_encoder.py
import torch
from torch.nn.functional import softmax
import pytorch_lightning as pl
from data.processed.vsr_dataset import VSRDataset
from data.processed.whatsup_dataset import WhatsUpDataset
from data.processed.biscor_dataset import BISCORDataset
import transformers
from utils.model_helpers import load_vision_model_components


class DualEncoder(pl.LightningModule):
    """
    Wrapper Lightning Module for dual-encoder models: fine-tuning / zero-shot evaluation.
    """
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()

         # --- Params ---
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps
        self.lr = args.lr
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2
        self.eps = args.eps
        self.weight_decay = args.weight_decay
        self.scheduler_off = args.scheduler_off
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.score = args.score

        print(f"args.gpus: {args.gpus}")

        # --- Load Model ---
        self.model_name = args.model
        self.model, self.config = load_vision_model_components(self.model_name)

    # -----------------------------
    # STEP (train/val/test)
    # -----------------------------
    def step(self, batch, split):
        inputs = self.move_to_device(batch, self.device)

        if split == "train":
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            logits = logits_per_image

            N = logits_per_image.shape[0]
            N_pairs = logits_per_image.shape[0] // 2
            ground_truth = torch.arange(N, device=self.device)        
            loss = 0.5 * (
                self.cross_entropy(logits_per_image, ground_truth) +
                self.cross_entropy(logits_per_text, ground_truth)
            )
            self.log(f'{split}_loss', loss, batch_size=N_pairs)
        else:
            if self.model_name == "pecore":
                image_features, text_features, logit_scale = self.model(inputs["image"], inputs["captions"])
                logits = logit_scale * image_features @ text_features.T
            else:
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image
        
        acc = 0 # Group score per each pair
        N_pairs = logits.shape[0] // 2
        for i in range(N_pairs):
            start = 2 * i
            end   = 2 * i + 2
            sub = logits[start:end, start:end]
            
            #        TexPos TexNeg
            # ImgPos   a      b
            # ImgNeg   c      d
            a, b = sub[0, 0], sub[0, 1] 
            c, d = sub[1, 0], sub[1, 1] 

            Tpos_2I = (a > c).item()
            Ipos_2T = (a > b).item()
            Tneg_2I = (d > b).item()
            Ineg_2T = (d > c).item()

            group_score = Ipos_2T and Ineg_2T and Tpos_2I and Tneg_2I
            acc += int(group_score)

        acc /= N_pairs

        # Logging
        self.log(f'{split}_accuracy', acc, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=N_pairs)
        return loss if split == 'train' else acc
    

    def move_to_device(self, batch, device):
        if self.model_name == "pecore":
            return {
                "image":    batch["image"].to(device),
                "captions": batch["captions"].to(device),
            }
        else:
            return {k: v.to(device) for k, v in batch.items()}
    
    # -----------------------------
    # LIGHTNING STEP METHODS
    # -----------------------------
    def training_step(self, batch, batch_idx):
        return self.step(batch, split="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, split="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, split="test")

    # -----------------------------
    # CONFIGURE OPTIMIZER
    # -----------------------------
    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            betas=(self.beta_1, self.beta_2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        if self.scheduler_off:
            return [optimizer]
        else:
            scheduler = {
                "scheduler": transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps),
                "interval": "step"
            }
            return [optimizer], [scheduler]