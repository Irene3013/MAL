#!/bin/bash

#SBATCH --job-name=pecore_ft                     # Name of the process
#SBATCH --cpus-per-task=2                      # Number of CPU cores (2 is reasonable)
#SBATCH --gres=gpu:1                           # Number of GPUs (usually light processes only need 1)
#SBATCH --mem=64G                              # RAM memory needed (8-16GB is reasonable for our servers, sometimes you'll need more)
#SBATCH --output=/gaueko0/users/ietxarri010/MAL/log_prompt.log
#SBATCH --error=/gaueko0/users/ietxarri010/MAL/error_prompt.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=irene.etxarri@ehu.eus

source /gaueko0/users/ietxarri010/env/pe_core/bin/activate

#export TRANSFORMERS_CACHE="/ncache/hub/"

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 2 --max_steps 5000 --accumulate_grad_batches 1 \
#    --run_name pecore_E5 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9


srun python main_whatsup_eval.py --model "pecore" \
   --ckpt "pecore-E5-v1-epoch=13-val_accuracy=1.00.ckpt"

srun python main_whatsup_eval.py --model "pecore" \
   --ckpt "pecore-E4-v2-epoch=04-val_accuracy=1.00.ckpt"

srun python main_whatsup_eval.py --model "pecore" \
   --ckpt "pecore-E4-v3-epoch=05-val_accuracy=1.00.ckpt"

srun python main_whatsup_eval.py --model "pecore" \
   --ckpt "pecore-E4-v4-epoch=06-val_accuracy=1.00.ckpt"

srun python main_whatsup_eval.py --model "pecore" \
   --ckpt "pecore-E4-v5-epoch=04-val_accuracy=1.00.ckpt"

srun python main_whatsup_eval.py --model "pecore" \
   --ckpt "pecore-E4-v6-epoch=07-val_accuracy=1.00.ckpt"

srun python main_whatsup_eval.py --model "pecore" \
   --ckpt "pecore-E5-v7-epoch=03-val_accuracy=1.00.ckpt"

srun python main_whatsup_eval.py --model "pecore" \
   --ckpt "pecore-E4-v8-epoch=27-val_accuracy=1.00.ckpt"

srun python main_whatsup_eval.py --model "pecore" \
   --ckpt "pecore-pecore_E5-v9-epoch=02-val_accuracy=1.00.ckpt"