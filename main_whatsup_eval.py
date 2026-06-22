import torch
import json
import os
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from itertools import combinations

def get_all_objects(dataset):
    """
    Extrae de forma eficiente todos los objetos únicos presentes en el dataset
    evitando duplicados y problemas con preposiciones largas.
    """
    unique_objects = set()
    
    for item in dataset:
        caption = item['caption_options'][0] 
        words = caption.strip(".").split()
        # get objects
        obj1 = words[1]
        obj2 = words[-1]
        # Add to set
        unique_objects.add(obj1)
        unique_objects.add(obj2)
    return sorted(list(unique_objects))



## Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Model's checkpoint to be loaded before training."
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="/gaueko0/users/ietxarri010/out_L/", help="Model's checkpoint path."
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
    parser.add_argument(
        "--output_name", type=str, required=True, help="Name for output file."
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
        self.data_path = Path(data_path) / "controlled_clevr_dataset_ordered.json"
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
    similarity = evaluate_object_similarity(model, processor, image_processor, tokenizer, dataset, device)
    return {
        'individual': individual, 
        'pairwise': pairwise, 
        'setwise': setwise,
        'similarity': similarity,
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

def score_caption_similarity(model, processor, image_processor, tokenizer, image, device, objects):
    captions = [f'a picture of a {obj}' for obj in objects]
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
    set_size = 4
    total_sets = len(dataset) // set_size
    correct_sets = 0

    for set_idx in range(total_sets):
        set_correct = True
        for i in range(set_size):
            item_idx = set_idx * set_size + i
            item = dataset[item_idx]
            pred_idx = score_image_captions(model, processor, image_processor, tokenizer, item["image"], item["caption_options"], device)
            if pred_idx != 0:
                set_correct = False
                break  # falla el set entero, no hace falta seguir

        if set_correct:
            correct_sets += 1

    accuracy = correct_sets / total_sets
    print(f"[Set-wise] Accuracy: {accuracy:.4f} ({correct_sets}/{total_sets})")
    return accuracy

def evaluate_pairwise(model, processor, image_processor, tokenizer, dataset, device):
    """
    Pair-wise: una pareja son todos los pares de imágenes que comparten el mismo par de objetos, pero
    con preposiciones contrarias (por ejemplo, right y left). La pareja se considera correcta solo si el modelo 
    acierta AMBAS imagenes de la pareja.
    """
    pair_size = 2
    total_pairs = len(dataset) // pair_size
    correct_pairs = 0

    for pair_idx in range(total_pairs):
        pair_correct = True
        for i in range(pair_size):
            item_idx = pair_idx * pair_size + i
            item = dataset[item_idx]
            pred_idx = score_image_captions(model, processor, image_processor, tokenizer, item["image"], item["caption_options"], device)
            if pred_idx != 0:
                pair_correct = False
                break  # falla el pair entero, no hace falta seguir
        if pair_correct:
            correct_pairs += 1

    accuracy = correct_pairs / total_pairs
    print(f"[Pair-wise] Accuracy: {accuracy:.4f} ({correct_pairs}/{total_pairs})")
    return accuracy

def evaluate_object_similarity(model, processor, image_processor, tokenizer, dataset, device):
    objects = get_all_objects(dataset)
    correct = 0
    total = len(dataset)

    for i in range(total):
        item = dataset[i]

        # get current image objects
        caption = item['caption_options'][0] 
        words = caption.strip(".").split()
        obj1 = words[1]
        obj2 = words[-1]
        
        # discard form objects set
        filtered_objects  = [obj for obj in objects if obj not in [obj1, obj2]]

        pred_idx1 = score_caption_similarity(model, processor, image_processor, tokenizer, item["image"], device, [obj1] + filtered_objects)
        pred_idx2 = score_caption_similarity(model, processor, image_processor, tokenizer, item["image"], device, [obj2] + filtered_objects)
        correct += 0.5 * ((pred_idx1 == 0) + (pred_idx2 == 0))

    accuracy = correct / total
    print(f"[Similarity] Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy    


def save_results(output_path, args, accuracies):
    results_file = Path(output_path) / f"{args.model}_{args.output_name}_results.json"
    
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

    # torch.serialization.add_safe_globals([argparse.Namespace])

    print("Parsing args...")
    args = parse_args()

    # Load model and processor 
    if args.model == "clip":
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-large-patch14"
        processor = CLIPProcessor.from_pretrained(model_name)

        if args.ckpt == None:
            model = CLIPModel.from_pretrained(model_name)
        else:
            #checkpoint = torch.load(f"{args.ckpt_path}/{args.ckpt}", map_location="cpu")
            checkpoint = torch.load(
                f"{args.ckpt_path}/{args.ckpt}",
                map_location="cpu",
                weights_only=False
            )
            state_dict = checkpoint["state_dict"]
            
            # prepare state dict to load (stripe ".model")
            stripped = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
            model = CLIPModel.from_pretrained(model_name)  # arquitectura base
            model.load_state_dict(stripped)
    else:
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as coreTransforms

        model_name = "PE-Core-L14-336"

        if args.ckpt == None:
            model = pe.CLIP.from_config(model_name, pretrained=True)
        else:
            #checkpoint = torch.load(f"{args.ckpt_path}/{args.ckpt}", map_location="cpu")
            checkpoint = torch.load(
                f"{args.ckpt_path}/{args.ckpt}",
                map_location="cpu",
                weights_only=False
            )
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

    #PRINT OBJECTS
    objects = get_all_objects(dataset)
    print(f"Total number of objects: {len(objects)}")
    print(objects)

if __name__ == "__main__":
    main_program()


