from transformers import CLIPModel, CLIPTokenizer
import torch
import csv
import os
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
        "--group_size", type=int, default=1, help="Precision for the GPUs."
    )
    parser.add_argument(
        "--experiment", type=str, help="Select dataset to be trained on."
    )
    parser.add_argument(
        "--variant", type=str, default=None, help="Select dataset variant to be trained on."
    )
    parser.add_argument(
        "--paraphrase", type=str, default=None, help="Model's checkpoint to be loaded before training."
    )
    args = parser.parse_args()
    return args


def get_text_embeddings(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.text_model(**inputs)
        embeddings = output.pooler_output  # (B, hidden_dim) - espacio textual puro
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
            checkpoint = torch.load(args.ckpt, map_location="cpu")
            print(checkpoint.keys())  
            print(list(checkpoint["state_dict"].keys())[:5])
            return 0
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
    
    
    if args.paraphrase is None: 

        seen = set()

        output_file = os.path.join(args.output_path, f"neg_similarity_{args.model}_{args.variant}.csv")
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["relation", "caption_pos", "caption_neg", "similarity"])
            writer.writeheader()

            # pos vs neg
            for sample in dataset:
                anchor   = sample["caption_pos"]
                hard_neg = sample["caption_neg"]
                relation = sample["relation"]

                # Saltar si ya procesamos este par
                pair_key = (anchor, hard_neg)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                
                embs = get_text_embeddings([anchor, hard_neg], model, tokenizer, device)
                similarity = (embs[0] @ embs[1]).item()
                
                # print(f"Anchor:    {anchor}")
                # print(f"Hard-neg:  {hard_neg}")
                # print(f"Similarity: {similarity:.4f}\n")

                writer.writerow({
                    "relation":    relation,
                    "caption_pos": anchor,
                    "caption_neg": hard_neg,
                    "similarity":  round(similarity, 4),
                })

    else:
        dataset2 = RELDataset(
            version = args.paraphrase,
            split="test",
            data_path=args.root,
            model=args.model,
            config=None
        )

        from itertools import product
    
        output_file = os.path.join(args.output_path, f"par_similarity_{args.model}_{args.variant}.csv")
        
        if args.group_size == 1:  # o args.group_size si quieres parametrizarlo

            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["relation", "caption_1", "caption_2", "similarity"])
                writer.writeheader()

                for sample, sample2 in zip(dataset, dataset2):
                    anchor   = sample["caption_pos"]
                    paraphrase = sample2["caption_pos"]
                    relation = sample["relation"]

                    embs = get_text_embeddings([anchor, paraphrase], model, tokenizer, device)
                    similarity = (embs[0] @ embs[1]).item()

                    # print(f"Anchor:      {anchor}")
                    # print(f"Paraphrase:  {paraphrase}")
                    # print(f"Similarity:  {similarity:.4f}\n")

                    writer.writerow({
                        "relation":    relation,
                        "caption_1": anchor,
                        "caption_2": paraphrase,
                        "similarity":  round(similarity, 4),
                    })
        else:
            # Agrupar ambos datasets en grupos de 3 (por imagen compartida)
            samples1 = list(dataset)
            samples2 = list(dataset2)

            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["relation", "template_1", "caption_1", "template_2", "caption_2", "similarity"])
                writer.writeheader()

                for i in range(0, len(samples1), args.group_size):
                    group1 = samples1[i:i + args.group_size]
                    group2 = samples2[i:i + args.group_size]

                    relation = group1[0]["relation"]

                    # Todas las combinaciones entre plantillas de v1 y v2
                    for (idx1, s1), (idx2, s2) in product(enumerate(group1), enumerate(group2)):
                        cap1 = s1["caption_pos"]
                        cap2 = s2["caption_pos"]

                        if cap1 == cap2:
                            continue

                        embs = get_text_embeddings([cap1, cap2], model, tokenizer, device)
                        similarity = (embs[0] @ embs[1]).item()

                        writer.writerow({
                            "relation":      relation,
                            "template_1":   idx1,        # 0, 1 o 2
                            "caption_1":    cap1,
                            "template_2":   idx2,        # 0, 1 o 2
                            "caption_2":    cap2,
                            "similarity":    round(similarity, 4),
                        })
        
    
if __name__ == "__main__":
    main_program()