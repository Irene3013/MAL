from transformers import CLIPModel, CLIPTokenizer
import torch
import csv
import os
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from itertools import product


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
        "--root", type=str, default="/gaueko0/users/ietxarri010/MAL/data/par", help="Path to the data files."
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
        "--group_size", type=int, default=1, help="Precision for the GPUs."
    )
    parser.add_argument(
        "--experiment", type=str, help="Select dataset to be trained on."
    )
    parser.add_argument(
        "--variant1", type=str, default=None, help="Select dataset variant to be trained on."
    )
    parser.add_argument(
        "--variant2", type=str, default=None, help="Select dataset variant to be trained on."
    )
    args = parser.parse_args()
    return args


class RELDataset(Dataset):
    def __init__(self, version="1", data_path="data"):
        self.data_path = Path(data_path) / f"paraphrase_{version}.csv"
        assert self.data_path.exists(), f"Root directory does not exist."
        self.dataset = self._load_csv()

    def _load_csv(self):
        with open(self.data_path, newline="\n", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            return list(reader)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pos_capt,neg_capt,relation,shape1,color1,shape2,color2,image = self.dataset[idx]
        return {
            "caption_pos": pos_capt,
            "caption_neg": neg_capt,
            "relation": relation
        }
        
    
def get_text_embeddings(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings

def main_program():

    torch.serialization.add_safe_globals([argparse.Namespace])

    print("Parsing args...")
    args = parse_args()


    # Load model and tokenizer 
    if args.model == "clip":
        model_name = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        if args.ckpt == None:
            model = CLIPModel.from_pretrained(model_name)
        else:
            checkpoint = torch.load(args.ckpt, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            
            # prepare state dict to load (stripe ".model")
            stripped = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
            
            model = CLIPModel.from_pretrained(model_name)  # arquitectura base
            model.load_state_dict(stripped)
    else:
        # TODO PE core
        raise NotImplementedError
    
    # Move to device
    model.eval()
    device = torch.device("cuda" if args.gpus > 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dataset1 = RELDataset(
            version = args.variant1,
            data_path=args.root,
    )
    
    dataset2 = RELDataset(
            version = args.variant2,
            data_path=args.root,
    )
    
    output_file = os.path.join(args.output_path, f"{args.model}_p{args.variant1}_p{args.variant2}.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relation", "caption_1", "caption_2", "similarity"])
        writer.writeheader()

        for sample1, sample2 in zip(dataset1, dataset2):
            pos_p1 = sample1["caption_pos"]
            neg_p1 = sample1["caption_neg"]
            pos_p2 = sample2["caption_pos"]
            relation = sample1["relation"]

            # similarity_1: pos vs neg
            embs = get_text_embeddings([pos_p1, neg_p1], model, tokenizer, device)
            similarity_1 = (embs[0] @ embs[1]).item()

            # similarity_2: pos1 vs pos2
            embs = get_text_embeddings([pos_p1, pos_p2], model, tokenizer, device)
            similarity_2 = (embs[0] @ embs[1]).item()

            score = 1 if similarity_2 > similarity_1 else 0

            writer.writerow({
                "relation":  relation,
                "score": score,
            })

    
if __name__ == "__main__":
    main_program()