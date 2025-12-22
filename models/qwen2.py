#models/qwen2.py
import torch
import string
import pytorch_lightning as pl
from project_datasets.vsr_dataset import VSRDataset
from project_datasets.whatsup_dataset import WhatsUpDataset
import transformers
from utils.model_helpers import load_vision_model_components

# from qwen_vl_utils import process_vision_info
# import core.vision_encoder.pe as pe
# import core.vision_encoder.transforms as coreTransforms

class Qwen2_VL(pl.LightningModule):
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
        self.score = "mc"#args.score

        print(f"args.gpus: {args.gpus}")
        #self.device = "cpu" if args.gpus == 0 else "cuda"

        # --- Load Model ---
        self.model, self.config = load_vision_model_components(args.model)
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
        # TODO: Implementar lógica de entrenamiento para Qwen2-VL
        return 0

    def eval_step(self, batch, split):
        inputs = batch["input"]
        labels = batch["label"]

        # Mover inputs al device
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # inputs = {k: v.squeeze(0) if v.dim() == 3 else v for k, v in inputs.items()}
        
        # Inference: Generation of the output
        # generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        # ]
        # output_text = self.config["processor"].batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        # acc = output_text == labels

        outputs = []
        for input in inputs:
            # Inputs to device
            input = {k: v.to(self.device) for k, v in input.items()}
            input = {k: v.squeeze(0) if v.dim() == 3 else v for k, v in input.items()}

            # Inference: Generation of the output
            generated_ids = self.model.generate(**input, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input['input_ids'], generated_ids)
            ]
            output_text = self.config["processor"].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = output_text[0].strip(string.punctuation)
            outputs.append([output_text])
        
        acc = 0
        for pred, gt in zip(output_text, labels):
            print(pred, gt)
            acc += (pred == gt) / len(inputs)
        print(outputs, labels)

        self.log(f'{split}_accuracy', acc, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
        return acc # Devolver la métrica de precisión


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
        