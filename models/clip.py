#models/clip.py
import torch
import clip
import pytorch_lightning as pl
from project_datasets.vsr_dataset import VSRDataset
import transformers


class ClipModel(pl.LightningModule):
    """
    Wrapper Lightning Module for CLIP model fine-tuning or zero-shot evaluation.
    """
    #def __init__(self, model_name="ViT-B/32", dataset="vsr", batch_size=8, lr=1e-5, device="cpu"): #TODO change to GPU
    def __init__(self, args): #TODO change to GPU
        super().__init__()

        self.save_hyperparameters()

         # --- Params ---
        self.model_name = args.target_model # Version of the CLIP model
        self.dataset = args.dataset         # [vsr, whatsup, biscor]
        self.batch_size = args.batch_size
        self.mode = args.clip_mode 

        if self.mode == "bin":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        ##self.loss_fn = torch.nn.CrossEntropyLoss()
        
        print(f"args.gpus: {args.gpus}")
        self.device_name = "cpu" if args.gpus == 0 else "cuda"


        # --- Validations ---
        available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64','ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        assert self.model_name in available_models, f"Unsupported clip model: '{self.model_name}'. Must be one of [{', '.join(available_models)}]."

        # --- Load CLIP ---
        self.model, self.preprocess = clip.load(self.model_name, device=self.device_name)

        # --- Load CLIP Tokenizer ---
        self.tokenizer = clip.tokenize

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
        - BCE if mode=bin
        - CE if mode!=bin
        """
        return self.loss_fn(logits, labels)

    # -----------------------------
    # FORWARD PASS
    # -----------------------------
    def forward(self, images, texts):
        """
        Forward pass through CLIP: 1 image + N caption.
        """
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)

        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        return logits
    
    def binary_forward(self, images, texts):
        """
        Forward pass through CLIP: 1 image + 1 caption.
        """
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)

        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #logits = (100.0 * image_features @ text_features.T).diagonal()
        logits = (image_features * text_features).sum(dim=1) * 100.0

        return logits


    # -----------------------------
    # STEP (train/val/test)
    # -----------------------------
    def step(self, batch, split):
        images, texts, labels = batch["image"], batch["text"], batch["label"]

        # Tokenize text and move tensors to device
        tokenized_texts = clip.tokenize(texts, truncate=True).to(self.device)
        images = images.to(self.device)

        # Forward pass
        if self.mode =="bin":
            logits = self.binary_forward(images, tokenized_texts)
    
            # Loss
            labels = labels.to(self.device).float()  # BCE needs float labels
            loss = self.compute_loss(logits, labels)

            # Accuracy
            accuracy = self.dataset_class.compute_accuracy(logits, labels, mode=self.mode)

        else: 
            logits = self.forward(images, tokenized_texts)  
            
            # Loss
            targets = [label.index(1) for label in labels] # CE needs correct labels index (targets)
            targets = torch.tensor(targets, dtype=torch.long).to(self.device)
            loss = self.compute_loss(logits, targets)

            # Accuracy
            accuracy = self.dataset_class.compute_accuracy(logits, targets, mode=self.mode)
        
        # Logging
        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
        self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
        
        return loss


    # Ensure labels on device and correct dtype --> what Cross Entropy Loss expects 
        # if labels is not None:
        #     labels = labels.to(self.device)
        #     if labels.dtype != torch.long:
        #         labels = labels.long()

    # # ======= Determine target for CrossEntropyLoss ======= # shape: (B, B?) or (B, C) depending on texts provided
        # # Case A: contrastive CLIP-style training where texts correspond 1:1 with images in batch
        # B = logits.size(0)
        # # If logits is square (B x B) -> typical CLIP similarity matrix, use arange targets
        # if logits.dim() == 2 and logits.size(1) == B:
        #     targets = torch.arange(B, device=self.device, dtype=torch.long)
        # else:
        #     # Otherwise assume labels from dataset are the class indices expected by the logits
        #     # and use them as targets, but validate range
        #     if labels is None:
        #         raise ValueError("labels is None but logits != square (B x B). Cannot infer targets.")
        #     targets = labels
        #     # Validate range to catch errors early
        #     n_classes = logits.size(1)
        #     if targets.min().item() < 0 or targets.max().item() >= n_classes:
        #         raise ValueError(f"Label values out of range for CrossEntropyLoss: labels min={targets.min().item()} max={targets.max().item()} n_classes={n_classes}")

        # Loss
        # loss = self.compute_loss(logits, targets)

        # Accuracy -- ensure metric gets CPU/GPU consistent tensors as expected by your dataset_class
        # Move targets to CPU if compute_accuracy expects CPU; otherwise pass device tensors.
        # try:
        #     accuracy = self.dataset_class.compute_accuracy(logits, targets)
        # except Exception:
        #     # fallback: compute a simple batch accuracy (argmax)
        #     preds = logits.argmax(dim=1)
        #     accuracy = (preds == targets).float().mean()


    # def step(self, batch, split):
    #     images, texts, labels = batch["image"], batch["text"], batch["label"]
        
    #     # Tokenize text
    #     tokenized_texts = clip.tokenize(texts, truncate=True).to(self.device)

    #     # Forward
    #     logits = self.forward(images.to(self.device), tokenized_texts)

    #     # Loss
    #     loss = self.compute_loss(logits, labels.to(self.device))
        
    #     # Accuracy -- depends on the dataset
    #     accuracy = self.dataset_class.compute_accuracy(logits, labels) 

    #     #Logging
    #     self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
    #     self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)

    #     return loss

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
