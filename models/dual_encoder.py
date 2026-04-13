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
        self.scheduler_off = args.scheduler_off
        self.batch_size = args.batch_size
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.score = args.score

        print(f"args.gpus: {args.gpus}")
        #self.device = "cpu" if args.gpus == 0 else "cuda"

        # --- Load Model ---
        self.model_name = args.model
        self.model, self.config = load_vision_model_components(self.model_name)

    # -----------------------------
    # STEP (train/val/test)
    # -----------------------------
    def step(self, batch, split):
        if split == "train":
            return self.train_step(batch, split)
        else:
            return self.eval_step(batch, split)

    def train_step(self, batch, split):
        inputs = self.move_to_device(batch, self.device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        ground_truth = torch.arange(2*self.batch_size, device=self.device)        
        loss = 0.5 * (
            self.cross_entropy(logits_per_image, ground_truth) +
            self.cross_entropy(logits_per_text, ground_truth)
        )

        self.log(f'{split}_loss', loss, batch_size=self.batch_size)
        return loss
      
    def move_to_device(self, batch, device):
        if self.model_name == "pecore":
            return {
                "image":    batch["image"].to(device),
                "captions": batch["captions"].to(device),
            }
        else:
            return {k: v.to(device) for k, v in batch.items()}
       
    def eval_step(self, batch, split): 
        inputs = self.move_to_device(batch, self.device)

        if self.model_name == "pecore":
            image_features, text_features, logit_scale = self.model(inputs["image"], inputs["captions"])
            logits = logit_scale * image_features @ text_features.T
        else:
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image
        
        acc = 0 # Accuracy per each pair
        for i in range(self.batch_size):
            start = 2 * i
            end   = 2 * i + 2
            sub = logits[start:end, start:end]

            a, b = sub[0, 0], sub[0, 1]
            c, d = sub[1, 0], sub[1, 1]

            Ipos_2T = (a > c).item()
            Ineg_2T = (d > b).item()
            Tpos_2I = (a > b).item()
            Tneg_2I = (d > c).item()

            group_score = Ipos_2T and Ineg_2T and Tpos_2I and Tneg_2I
            acc += int(group_score)

        acc /= self.batch_size

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

    # -----------------------------
    # CONFIGURE OPTIMIZER
    # -----------------------------
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

    # def train_step(self, batch, split):
    #     images = batch["image"].to(self.device)
    #     texts  = batch["text"].to(self.device)

    #     outputs = self.model(texts, images) 
    #     logits_per_image, logits_per_text  = outputs.logits_per_image, outputs.logits_per_text

    #     batch_size = images.size(0)
    #     ground_truth = torch.arange(batch_size, device=self.device, dtype=torch.long) # Every caption is true

    #     loss = 0.5 * (
    #         self.cross_entropy(logits_per_image, ground_truth) +
    #         self.cross_entropy(logits_per_text,  ground_truth)
    #     )

    #     # Logging
    #     self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
    #     return loss


    # def eval_step(self, batch, split):
    #     labels = batch["label"].to(self.device)
    #     inputs_list = batch["input"]

    #     # Forward pass each input
    #     logits_list = []
    #     for inputs in inputs_list:

    #         if self.model_name == "pecore":
    #             image = inputs['image'].to(self.device)
    #             captions = inputs['captions'].to(self.device)
    #             image_features, text_features, logit_scale = self.model(image, captions)
    #             I2T_logits = logit_scale * image_features @ text_features.T

    #         else:
    #             inputs = inputs.to(self.device)
    #             outputs = self.model(**inputs)
    #             I2T_logits = outputs.logits_per_image

    #         logits_list.append(I2T_logits)

    #     logits = torch.cat(logits_list, dim=0)
    #     acc = self.compute_accuracy(logits, labels, self.score)

    #     # Logging
    #     self.log(f'{split}_accuracy', acc, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
    #     return acc