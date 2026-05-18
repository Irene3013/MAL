from transformers import CLIPModel, CLIPProcessor
import torch
import json
import os
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image



## Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Model's checkpoint to be loaded before training."
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="/gaueko0/users/ietxarri010/out/", help="Model's checkpoint path."
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs in use. (0 == cpu)"
    )
    parser.add_argument(
        "--root", type=str, default="/gaueko0/users/ietxarri010/MAL/data/raw/whatsup", help="Path to the data files."
    )
    parser.add_argument(
        "--output_path", type=str, default="/gaueko0/users/ietxarri010/MAL/", help="Output directory for plots and models."
    )

    # Model args
    parser.add_argument(
        "--model", type=str, required=True, choices=["clip", "siglip", "siglip2", "pecore", "qwen2", "clip-flant5"],
        help = "Model type to be fine-tuned."
    )
    parser.add_argument(
        "--score", type=str, required=True, choices=["individual", "set-wise"],
        help = "Method to compute score."
    )
    args = parser.parse_args()
    return args



class WhatsupDataset(Dataset):
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path) / "controlled_clevr_dataset.json"
        self.image_path = Path(data_path) / "controlled_clevr"
        self.dataset = self._load_json()

    def _load_json(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    def _load_image(self, orig_path):
        img_path = self.image_path / orig_path.split("/")[-1]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "image": self._load_image(item["image_path"]),
            "caption_options": item["caption_options"],
            "correct_option": item["caption_options"][0], # The first option is the correct one
        }
    

def evaluate(model, processor, dataset, device, score_mode):
    if score_mode == "individual":
        return evaluate_individual(model, processor, dataset, device)
    elif score_mode == "set-wise":
        return evaluate_setwise(model, processor, dataset, device)


def score_image_captions(model, processor, image, captions, device):
    """Devuelve el índice de la caption con mayor similitud con la imagen."""
    inputs = processor(
        text=captions,
        images=[image] * len(captions),
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image: (1, num_captions)
        probs = F.softmax(outputs.logits_per_image, dim=-1)

    return probs.argmax().item()


def evaluate_individual(model, processor, dataset, device):
    correct = 0
    total = len(dataset)

    for i in range(total):
        item = dataset[i]
        pred_idx = score_image_captions(model, processor, item["image"], item["caption_options"], device)
        if pred_idx == 0:  # la opción 0 es siempre la correcta
            correct += 1

    accuracy = correct / total
    print(f"[Individual] Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def evaluate_setwise(model, processor, dataset, device):
    """
    Set-wise: un set son todas las imágenes que comparten el mismo par de objetos
    pero varían la preposición. El set se considera correcto solo si el modelo
    acierta TODAS las imágenes del set.
    
    Asumimos que los items están ordenados en grupos de 4 en el JSON
    (tal como está estructurado el dataset de whatsup).
    """
    set_size = 4
    total_sets = len(dataset) // set_size
    correct_sets = 0

    for set_idx in range(total_sets):
        set_correct = True
        for i in range(set_size):
            item_idx = set_idx * set_size + i
            item = dataset[item_idx]
            pred_idx = score_image_captions(model, processor, item["image"], item["caption_options"], device)
            if pred_idx != 0:
                set_correct = False
                break  # falla el set entero, no hace falta seguir

        if set_correct:
            correct_sets += 1

    accuracy = correct_sets / total_sets
    print(f"[Set-wise] Accuracy: {accuracy:.4f} ({correct_sets}/{total_sets})")
    return accuracy


def save_results(output_path, args, accuracy):
    results_file = Path(output_path) / "results.json"
    
    # Cargar resultados anteriores si existen
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = []

    # Añadir nueva entrada
    results.append({
        "model": args.model,
        "ckpt": args.ckpt,
        "score": args.score,
        "accuracy": accuracy,
    })

    os.makedirs(output_path, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_file}")

def main_program():

    torch.serialization.add_safe_globals([argparse.Namespace])

    print("Parsing args...")
    args = parse_args()

    # Load model and processor 
    if args.model == "clip":
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)

        if args.ckpt == None:
            model = CLIPModel.from_pretrained(model_name)
        else:
            checkpoint = torch.load(f"{args.ckpt_path}/{args.ckpt}", map_location="cpu")
            state_dict = checkpoint["state_dict"]
            
            # prepare state dict to load (stripe ".model")
            stripped = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
            
            model = CLIPModel.from_pretrained(model_name)  # arquitectura base
            model.load_state_dict(stripped)
    else:
        raise NotImplementedError
    
    # Move to device
    model.eval()
    device = torch.device("cuda" if args.gpus > 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = WhatsupDataset(data_path=args.root)

    print(f"Evaluating with '{args.score}' scoring on {len(dataset)} samples...")
    accuracy = evaluate(model, processor, dataset, device, score_mode=args.score)
    save_results(args.output_path, args, accuracy)

if __name__ == "__main__":
    main_program()


