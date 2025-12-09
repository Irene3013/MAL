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
# import core.vision_encoder.pe as pe
# import core.vision_encoder.transforms as coreTransforms

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
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.confifg = {
                "processor": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
                "transform": None,
                "tokenizer": None,
                "params": {"padding": True, "truncation": True}
            }

        elif args.model == "siglip":
            self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
            self.confifg = {
                "processor": AutoProcessor.from_pretrained("google/siglip-base-patch16-224"),
                "transform": self.preprocess, # crop images
                "tokenizer": None,
                "params": {"padding": "max_length", "max_length": 64}
            }

        elif args.model == "siglip2":
            self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
            self.confifg = {
                "processor": AutoProcessor.from_pretrained("google/siglip2-base-patch16-224"),
                "transform": self.preprocess, # crop images
                "tokenizer": None,
                "params": {"padding": "max_length", "max_length": 64}
            }

        elif args.model == "PEcore":
            # TODO https://huggingface.co/facebook/PE-Core-B16-224
            # self.model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True) 
            # self.model = self.model.cuda()
            # self.confifg = {
            #     "processor": coreTransforms.get_image_transform(self.model.image_size),
            #     "transform": self.preprocess, # crop images
            #     "tokenizer": coreTransforms.get_text_tokenizer(self.model.context_length),
            #     "params": None
            # }
            raise NotImplementedError

        else:
            raise NotImplementedError

        # --- Accuracy depending on dataset ---
        if args.dataset == "whatsup":
            # precision / pair-wise / set-wise 
            self.compute_accuracy = WhatsUpDataset.compute_accuracy
        else: 
            # precision 
            self.compute_accuracy = VSRDataset.compute_accuracy        

    # -----------------------------
    # COMPUTE LOSS
    # -----------------------------
    def compute_loss(self, logits, labels):
        """
        Compute discriminative loss for CLIP.
        """
        return self.cross_entropy(logits, labels)
    
    def compute_contrastive_loss(self, logits, labels):
        """
        Compute contrastive loss for CLIP.
        """
        T2I_logits = logits
        I2T_logits = logits.T
        return (self.cross_entropy(T2I_logits, labels) + self.cross_entropy(I2T_logits, labels)) / 2.0

    # -----------------------------
    # STEP (train/val/test)
    # -----------------------------
    def step(self, batch, split):
        labels = batch["label"].to(self.device)
        inputs_list = batch["input"]
        
        outputs = self.model(**inputs_list)
        loss = 0.5 * (self.cross_entropy(outputs.logits_per_image) + self.cross_entropy(outputs.logits_per_text)) 
        # T2I_logits_list = []
        # I2T_logits_list = []
        
        # # Forward pass each input
        # for inputs in inputs_list:     
        #     inputs = inputs.to(self.device)
        #     outputs = self.model(**inputs)
        #     T2I_logits_list.append(outputs.logits_per_image)
        #     I2T_logits_list.append(outputs.logits_per_text)

        # Loss and accuracy
        #logits = torch.cat(T2I_logits_list, dim=0) 
        #loss = self.compute_loss(logits, labels)
        #accuracy = self.compute_accuracy(logits, labels, self.score)
        
        accuracy = self.compute_accuracy(outputs.logits_per_image, labels, self.score)
        
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