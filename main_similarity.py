from transformers import CLIPModel, CLIPTokenizer
import torch
import argparse
import torch.nn.functional as F
from data.processed.relations_dataset import RELDataset


## Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Model's checkpoint to be loaded before training."
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs in use. (0 == cpu)"
    )
    parser.add_argument(
        "--root", type=str, default="/gaueko0/users/ietxarri010/MAL/data", help="Path to the data files."
    )
    parser.add_argument(
        "--image_path", type=str, default="/gaueko0/users/ietxarri010/MAL/data", help="Path to the image files if its different from the annotations files."
    )
    parser.add_argument(
        "--output_path", type=str, default="/gaueko0/users/ietxarri010/out/", help="Output directory for plots and models."
    )

    # Model args
    parser.add_argument(
        "--model", type=str, required=True, choices=["clip", "siglip", "siglip2", "pecore", "qwen2", "clip-flant5"],
        help = "Model type to be fine-tuned."
    )
    
    # DataLoader args
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["vsr", "whatsup", "cocospatial", "gqaspatial", "biscor", "rel"], help="Select dataset to be trained on."
    )
    parser.add_argument(
        "--experiment", type=str, help="Select dataset to be trained on."
    )
    parser.add_argument(
        "--variant", type=str, default=None, help="Select dataset variant to be trained on."
    )
    args = parser.parse_args()
    return args


def get_text_embeddings(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # <- esto
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings



def main_program():

    torch.serialization.add_safe_globals([argparse.Namespace])

    print("Parsing args...")
    args = parse_args()


    # Cargar modelo y tokenizer base
    if args.model == "clip":
        if args.ckpt == None:
            model_name = "openai/clip-vit-base-patch32"
            tokenizer = CLIPTokenizer.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            model.eval()
        else:
           raise NotImplementedError
    else:
        raise NotImplementedError
    
    # Move to device
    device = torch.device("cuda" if args.gpus > 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Tu dataset de pares (ejemplo, hard-negative)
    dataset = RELDataset(
            version = args.variant,
            split="test",
            data_path=args.root,
            model=args.model,
            config=None
    )
    
    for sample in dataset:
        anchor   = sample["caption_pos"]
        hard_neg = sample["caption_neg"]
        
        embs = get_text_embeddings([anchor, hard_neg], model, tokenizer, device)
        similarity = (embs[0] @ embs[1]).item()
        
        print(f"Anchor:    {anchor}")
        print(f"Hard-neg:  {hard_neg}")
        print(f"Similarity: {similarity:.4f}\n")
if __name__ == "__main__":
    main_program()