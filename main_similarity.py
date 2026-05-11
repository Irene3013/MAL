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
    # parser.add_argument(
    #     "--num_workers", type=int, default=2, help="Workers used in the dataloader." 
    # )

    # # Trainer args
    # parser.add_argument(
    #     "--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps. (1 == do not use gradient accumulation)"
    # )
    # parser.add_argument(
    #     "--scheduler_off", action="store_true", help="Do not use any scheduler"
    # )
    # parser.add_argument(
    #     "--deepspeed", action="store_true", help="Use deepspeed stage-2 offload strategy."
    # )
    # parser.add_argument(
    #     "--val_check_interval", type=float, default=1.0, help="How often within a training epoch to check the val set. (1.0 == every epoch)"
    # )
    # parser.add_argument(
    #     "--lr", type=float, default=1e-6, help="Learning rate."
    # )
    # parser.add_argument(
    #     '--beta-1',       type=float, default=0.9
    # )
    # parser.add_argument(
    #     '--beta-2',       type=float, default=0.98
    # )
    # parser.add_argument(
    #     '--eps',          type=float, default=1e-6
    # )
    # parser.add_argument(
    #     '--weight-decay', type=float, default=0.1
    # )
    # parser.add_argument(
    #     "--precision", type=int, default=32, choices=[16, 32, 64], help="Precision for the GPUs."
    # )
    # parser.add_argument(
    #     "--warmup_steps", type=int, default=50, help="Warmup steps to be done during training."
    # )
    # parser.add_argument(
    #     "--use_epochs", action="store_true", help="Use max_epoch for training duration."
    # )
    # parser.add_argument(
    #     "--max_steps", type=int, default=10000, help="Steps to be done during training."
    # )
    # parser.add_argument(
    #     "--max_epochs", type=int, default=10, help="Epochs to be done during training."
    # )
    # parser.add_argument(
    #     "--seed", type=int, default=-1, help="Seed."
    # )
    
    args = parser.parse_args()
    return args


def get_text_embeddings(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    # Normalizar (importante para cosine similarity)
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings



def main_program():

    torch.serialization.add_safe_globals([argparse.Namespace])

    print("Parsing args...")
    args = parse_args()


    # Cargar modelo y tokenizer base
    if args.model == "clip":
        if args.checkpoint == None:
            model_name = "openai/clip-vit-base-patch32"
        else:
            model_name = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model.eval()
    else:
        raise NotImplementedError
    
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
        
        embs = get_text_embeddings([anchor, hard_neg], model, tokenizer)
        similarity = (embs[0] @ embs[1]).item()
        
        print(f"Anchor:    {anchor}")
        print(f"Hard-neg:  {hard_neg}")
        print(f"Similarity: {similarity:.4f}\n")
if __name__ == "__main__":
    main_program()