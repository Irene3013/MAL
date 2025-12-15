#models/qwen2.py
import torch
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
        self.score = args.score

        print(f"args.gpus: {args.gpus}")
        self.device_name = "cpu" if args.gpus == 0 else "cuda"

        # --- Load Model ---
        self.model, self.config = load_vision_model_components(args.model, self.device_name)
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
        
        labels = batch["label"]
        inputs = batch["input"]

        # Asegúrate de que todos los tensores están en el dispositivo correcto
        # Los `inputs` de Qwen2 son un diccionario de tensores.
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                 inputs[key] = inputs[key].to(self.device) # Asumiendo self.device existe
            
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.config["processor"].batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # TODO: Implementar el cálculo de la precisión (acc) comparando output_text con labels
        
        print(output_text)
        print(labels)
        return 0 # Devolver la métrica de precisión


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
        
# # CLIP image processor
# self.preprocess = transforms.Compose([
#     Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
#     CenterCrop(224),
# ])

# # https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct 
# self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#   "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto"
# )

# self.model.to(self.device)
# self.config = {
#     "processor": AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct"),
#     "transform": None, #self.preprocess, # crop images for comparable results
#     "tokenizer": None,
#     "params": None
# }

# self.model_name = args.model