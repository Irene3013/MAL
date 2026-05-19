import torch
import json
import os
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from itertools import combinations

OPPOSITES = [{"left_of", "right_of"}, {"in-front_of", "behind_of"}]

def parse_image_name(image_path):
    """Extrae (obj1, relation, obj2) del nombre del fichero."""
    name = Path(image_path).stem  # e.g. "mug_right_of_knife"
    # Buscar qué relación contiene
    for rel in ["left", "right", "front", "behind"]:
        pattern = f"_{rel}_"
        if pattern in name:
            parts = name.split(pattern)
            obj1 = parts[0]           # "mug"
            obj2 = parts[1]           # "knife"
            return obj1, rel, obj2
    return None

def get_object_key(obj1, obj2):
    """Clave canónica para un par de objetos (orden alfabético)."""
    return tuple(sorted([obj1, obj2]))

def group_dataset(dataset):
    """
    Devuelve:
      - sets:  dict { (obj1, obj2) -> [item, item, item, item] }  (4 relaciones)
      - pairs: dict { (obj1, obj2, frozenset(rel_pair)) -> [item, item] }
    """
    from collections import defaultdict
    
    sets  = defaultdict(list)
    pairs = defaultdict(list)

    for item in dataset:
        parsed = parse_image_name(item["image_path"])
        if parsed is None:
            continue
        obj1, rel, obj2 = parsed
        obj_key = get_object_key(obj1, obj2)

        # Agrupar sets
        sets[obj_key].append(item)

        # Agrupar pares según relaciones opuestas
        for opposite_pair in OPPOSITES:
            if rel in opposite_pair:
                pair_key = (obj_key, frozenset(opposite_pair))
                pairs[pair_key].append(item)
                break

    return sets, pairs


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
    

def evaluate(model, processor, image_processor, tokenizer, dataset, device):
    individual = evaluate_individual(model, processor, image_processor, tokenizer, dataset, device)
    pairwise = evaluate_pairwise(model, processor, image_processor, tokenizer, dataset, device)
    setwise = evaluate_setwise(model, processor, image_processor, tokenizer, dataset, device)
    return {
        'individual': individual, 
        'pairwise': pairwise, 
        'setwise': setwise,
    }


def score_image_captions(model, processor, image_processor, tokenizer, image, captions, device):
    """Devuelve el índice de la caption con mayor similitud con la imagen."""
    if processor is not None:
        inputs = processor(
            text=captions,
            images=[image] * len(captions),
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits_per_image, dim=-1)
    
    else:
        # Procesar texto e imagen separados
        image_input = image_processor(image).unsqueeze(0).to(device)          # (1, C, H, W)
        text_input  = tokenizer(captions).to(device)                           # (N, seq_len)

        with torch.no_grad():
            image_features, text_features, logit_scale = model(image_input, text_input)
            logits = (logit_scale * image_features @ text_features.T).squeeze(0)  # (N,)
            probs  = F.softmax(logits, dim=-1)
    return probs.argmax().item()


def evaluate_individual(model, processor, image_processor, tokenizer, dataset, device):
    correct = 0
    total = len(dataset)

    for i in range(total):
        item = dataset[i]
        pred_idx = score_image_captions(model, processor, image_processor, tokenizer, item["image"], item["caption_options"], device)
        if pred_idx == 0:  # la opción 0 es siempre la correcta
            correct += 1

    accuracy = correct / total
    print(f"[Individual] Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def evaluate_setwise(model, processor, image_processor, tokenizer, dataset, device):
    """
    Set-wise: un set son todas las imágenes que comparten el mismo par de objetos
    pero varían la preposición. El set se considera correcto solo si el modelo
    acierta TODAS las imágenes del set.
    """
    sets, _ = group_dataset(dataset)
    correct_sets = 0
    total_sets = 0

    for obj_key, items in sets.items():
        if len(items) != 4:  # set incompleto, lo saltamos
            continue
        total_sets += 1
        set_correct = all(
            score_image_captions(model, processor, image_processor, tokenizer,
                                  item["image"], item["caption_options"], device) == 0
            for item in items
        )
        if set_correct:
            correct_sets += 1

    accuracy = correct_sets / total_sets
    print(f"[Set-wise] Accuracy: {accuracy:.4f} ({correct_sets}/{total_sets})")
    return accuracy
    # set_size = 4
    # total_sets = len(dataset) // set_size
    # correct_sets = 0

    # for set_idx in range(total_sets):
    #     set_correct = True
    #     for i in range(set_size):
    #         item_idx = set_idx * set_size + i
    #         item = dataset[item_idx]
    #         pred_idx = score_image_captions(model, processor, image_processor, tokenizer, item["image"], item["caption_options"], device)
    #         if pred_idx != 0:
    #             set_correct = False
    #             break  # falla el set entero, no hace falta seguir

    #     if set_correct:
    #         correct_sets += 1

    # accuracy = correct_sets / total_sets
    # print(f"[Set-wise] Accuracy: {accuracy:.4f} ({correct_sets}/{total_sets})")
    # return accuracy

def evaluate_pairwise(model, processor, image_processor, tokenizer, dataset, device):
    """
    Pair-wise: una pareja son todos los pares de imágenes que comparten el mismo par de objetos, pero
    con preposiciones contrarias (por ejemplo, right y left). La pareja se considera correcta solo si el modelo 
    acierta AMBAS imagenes de la pareja.
    """
    _, pairs = group_dataset(dataset)
    correct_pairs = 0
    total_pairs = 0

    for pair_key, items in pairs.items():
        if len(items) != 2:  # par incompleto, lo saltamos
            continue
        total_pairs += 1
        pair_correct = all(
            score_image_captions(model, processor, image_processor, tokenizer,
                                  item["image"], item["caption_options"], device) == 0
            for item in items
        )
        if pair_correct:
            correct_pairs += 1

    accuracy = correct_pairs / total_pairs
    print(f"[Pair-wise] Accuracy: {accuracy:.4f} ({correct_pairs}/{total_pairs})")
    return accuracy

    # pair_size = 2
    # total_pairs = len(dataset) // pair_size
    # correct_sets = 0

    # for set_idx in range(total_pairs):
    #     set_correct = True
    #     for i in range(pair_size):
    #         item_idx = set_idx * pair_size + i
    #         item = dataset[item_idx]
    #         pred_idx = score_image_captions(model, processor, image_processor, tokenizer, item["image"], item["caption_options"], device)
    #         if pred_idx != 0:
    #             set_correct = False
    #             break  # falla el pair entero, no hace falta seguir

    #     if set_correct:
    #         correct_sets += 1

    # accuracy = correct_sets / total_pairs
    # print(f"[Pair-wise] Accuracy: {accuracy:.4f} ({correct_sets}/{total_pairs})")
    # return accuracy


def save_results(output_path, args, accuracies):
    results_file = Path(output_path) / "results.json"
    
    # Cargar resultados anteriores si existen
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = []

    # Añadir nueva entrada
    for key, score in accuracies.items():
        results.append({
            "model": args.model,
            "ckpt": args.ckpt,
            "score": key,
            "accuracy": score,
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
        from transformers import CLIPModel, CLIPProcessor

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
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as coreTransforms

        model_name = "PE-Core-B16-224"

        if args.ckpt == None:
            model = pe.CLIP.from_config(model_name, pretrained=True)
        else:
            checkpoint = torch.load(f"{args.ckpt_path}/{args.ckpt}", map_location="cpu")
            state_dict = checkpoint["state_dict"]

            # prepare state dict to load (stripe ".model")
            stripped = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
            model = pe.CLIP.from_config(model_name, pretrained=True)  # arquitectura base
            model.load_state_dict(stripped)

        # Load image processor and tokenizer
        image_processor = coreTransforms.get_image_transform(model.image_size)
        tokenizer = coreTransforms.get_text_tokenizer(model.context_length)
    
    # Move to device
    model.eval()
    device = torch.device("cuda" if args.gpus > 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = WhatsupDataset(data_path=args.root)
    if args.model == "clip":
        accuracies = evaluate(model, processor, None, None, dataset, device)
    else:
        accuracies = evaluate(model, None, image_processor, tokenizer, dataset, device)

    save_results(args.output_path, args, accuracies)

if __name__ == "__main__":
    main_program()


