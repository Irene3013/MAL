import json
import os
from pathlib import Path
from PIL import Image

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn.functional import normalize
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import wandb

import sys
# Get the path to test-script (two levels up from main.py)
test_script_path = Path(__file__).resolve().parent.parent
sys.path.append(str(test_script_path))
from utils import load_model,  random_seed, dataset, count_parameters,  DATASETS, MODELS
from scheduler import cosine_lr

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,        default=10) 
    parser.add_argument('--device',                 default=None)
    parser.add_argument("--seed", type=int,         default=42)
    parser.add_argument('--model',                  default="PE-Core-B16-224", choices=list(MODELS.keys()), help="Select a model from the predefined list.")
    parser.add_argument('--checkpoint',type=Path,   default=None, help="Model checkpoint, e.g. CLIP_TROHN-Img.")
    parser.add_argument('--save-path', type=Path,   default='/gaueko0/users/imiranda014/GSCRATCH/Data/inference-time/crossmodal_ckpt')
    parser.add_argument('--dataset',                default='COCO')
    parser.add_argument('--batch-size', type=int,   default=400)
    parser.add_argument('--lr',                     default=1e-6)
    parser.add_argument('--beta-1',                 default=0.9)
    parser.add_argument('--beta-2',                 default=0.98)
    parser.add_argument('--eps',                    default=1e-6)
    parser.add_argument('--weight-decay',           default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='cosine_lr')
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument("--project_name", type=str, default='PE_FT')
    parser.add_argument("--run_name", type=str,     default='PE')

    args = parser.parse_args()

    os.system('wandb login')

    config = {"model":args.model,
            "learning_rate": args.lr, "beta1": args.beta_1, "beta2": args.beta_2, "epsilon": args.eps, "weight-decay": args.weight_decay,
            "epochs": args.epoch, "batch_size": args.batch_size}  
    run_name = f"{args.run_name}_FT_{args.dataset}"
    wandb.init(project = args.project_name, name = run_name, config=config)

    random_seed(seed = args.seed)

    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCH = args.epoch
    args.save_path.mkdir(exist_ok=True)

    model, preprocess, seq_length, tokenizer = load_model(args)
    print(f'There are #{count_parameters(model)} trainable parameters.')

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]


    optimizer = AdamW(
       [{"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": args.weight_decay}], 
        lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps)
    
    if 'COCO' in args.dataset:
        dataframe, dataset = dataset(args)
        dataset_size = len(dataset)  # total number of samples
        train_percent = 0.8           # 80% for training
        train_size = int(train_percent * dataset_size)  # number of training samples
        test_size = dataset_size - train_size  
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    elif 'train_TROHN_IMG' in args.dataset:
        train_dataframe, train_dataset = dataset(args)
        args.dataset = 'val_TROHN_IMG'
        val_dataframe, val_dataset = dataset(args)
        args.dataset = 'train_TROHN_IMG'
        train_size = len(train_dataset)  # total number of samples

    print(f'Train dataset: {len(train_dataset)} instances, Validation dataset: {len(val_dataset)}')
    train_dataloader = DataLoader(train_dataset, collate_fn = lambda x: x, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, collate_fn = lambda x: x, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2) 

    num_batches = len(train_dataloader)
    total_steps = num_batches * args.epoch
    if args.lr_scheduler == "cosine_lr":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup_steps, total_steps)
    
    loss_img = CrossEntropyLoss()
    loss_txt = CrossEntropyLoss()

    LAST_EPOCH = 0

    print(f'Fine-tuning for #{EPOCH} epochs.')
    best = 0
    best_epoch = 0
    for epoch in range(LAST_EPOCH, LAST_EPOCH+EPOCH):
        model.train()
        train_losses = []
        train_accuracy = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        i_accum=0
        for batch in pbar:
            all_images = [item['image'] for item in batch]
            all_captions = [item['caption'] for item in batch]
            if 'train_TROHN_IMG' in args.dataset:
                all_neg_images = [item['negative_image'] for item in batch]
                all_neg_captions = [item['negative_caption'] for item in batch]
                all_images = all_images + all_neg_images
                all_captions = all_captions + all_neg_captions
            n_images = len(all_images)
            step = num_batches * epoch + i_accum
            i_accum += 1
            if args.lr_scheduler == "cosine_lr":
                scheduler(step)
            ground_truth = torch.arange(n_images, dtype=torch.long, device=args.device)
            caption = tokenizer(all_captions).to(args.device)
            image =  torch.stack([preprocess(image) for image in all_images]).to(args.device)
            image_features, text_features, logit_scale = model(image, caption)
            logits_im = (logit_scale * image_features @ text_features.T)
            logits_text = logits_im.T
 
            loss = (loss_img(logits_im, ground_truth) + loss_txt(logits_text, ground_truth))/2
            acc_text =  (softmax(logits_text, dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
            acc_im =  (softmax(logits_im, dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
            acc = (acc_im + acc_text)/2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Train cross loss': loss.item(), 'Train cross acc': acc.item()})
            train_losses.append(loss.item())
            train_accuracy.append(acc.item())
        
        # Validation
        model.eval() 
    
        val_losses = [] 
        val_accuracy = []
        
        with torch.inference_mode():
            pbar = tqdm(val_dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                all_images = [item['image'] for item in batch]
                all_captions = [item['caption'] for item in batch]
                if 'train_TROHN_IMG' in args.dataset:
                    all_neg_images = [item['negative_image'] for item in batch]
                    all_neg_captions = [item['negative_caption'] for item in batch]
                    all_images = all_images + all_neg_images
                    all_captions = all_captions + all_neg_captions
                n_images = len(all_images)
                ground_truth = torch.arange(n_images, dtype=torch.long, device=args.device)
                caption = tokenizer(all_captions).to(args.device)
                image =  torch.stack([preprocess(image) for image in all_images]).to(args.device)
                image_features, text_features, logit_scale = model(image, caption)
                logits_im = (logit_scale * image_features @ text_features.T)
                logits_text = logits_im.T
              
                val_loss = (loss_img(logits_im, ground_truth) + loss_txt(logits_text, ground_truth))/2
                acc_text =  (softmax(logits_text, dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
                acc_im =  (softmax(logits_im, dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
                val_acc = (acc_im + acc_text)/2
                pbar.set_postfix({'Validation cross loss': val_loss.item(), 'Validation cross acc': val_acc.item()})
                val_losses.append(val_loss.item())
                val_accuracy.append(val_acc.item())
        wandb.log({'Train Cross-accuracy': (np.mean(train_accuracy)).item(), 'Train Cross-loss': (np.mean(train_losses)).item(), 'Validation Cross-accuracy' : (np.mean(val_accuracy)).item(), 'Validation Cross-loss': (np.mean(val_losses)).item()})        
        print(f'Epoch {epoch} - Train Cross-accuracy: {np.mean(train_accuracy):.4} - Train Cross-loss: {np.mean(train_losses):.4} - Validation Cross-accuracy: {np.mean(val_accuracy):.4} - Validation Cross-loss: {np.mean(val_losses):.4}' )
        if (np.mean(val_accuracy)).item() > best:  #
            try: 
                os.remove(args.save_path / f'best_{args.run_name}_{best_epoch}.pt') 
            except OSError:
                pass
            best = (np.mean(val_accuracy)).item()
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                },
                args.save_path / f'best_{run_name}_{epoch}.pt'
            )
            best_epoch = epoch 