#!/bin/bash

#SBATCH --job-name=clip_ft                     # Name of the process
#SBATCH --cpus-per-task=2                      # Number of CPU cores (2 is reasonable)
#SBATCH --gres=gpu:1                           # Number of GPUs (usually light processes only need 1)
#SBATCH --mem=64G                              # RAM memory needed (8-16GB is reasonable for our servers, sometimes you'll need more)
#SBATCH --output=/gaueko0/users/ietxarri010/MAL/log_prompt.log
#SBATCH --error=/gaueko0/users/ietxarri010/MAL/error_prompt.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=irene.etxarri@ehu.eus

source /gaueko0/users/ietxarri010/env/nire_env/bin/activate

#export TRANSFORMERS_CACHE="/ncache/hub/"

# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 200 --max_steps 10000 --accumulate_grad_batches 1 \
#    --run_name E9 --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v1


srun python main_whatsup_eval.py --model "clip" \
   --ckpt "clip-E6-v1-epoch=10-val_accuracy=0.99.ckpt"

srun python main_whatsup_eval.py --model "clip" \
   --ckpt "clip-E6-v2-epoch=12-val_accuracy=0.98.ckpt"

srun python main_whatsup_eval.py --model "clip" \
   --ckpt "clip-E6-v3-epoch=06-val_accuracy=0.99.ckpt"

srun python main_whatsup_eval.py --model "clip" \
   --ckpt "clip-E6-v4-epoch=04-val_accuracy=1.00.ckpt"

srun python main_whatsup_eval.py --model "clip" \
   --ckpt "clip-E6-v5-epoch=05-val_accuracy=1.00.ckpt"

srun python main_whatsup_eval.py --model "clip" \
   --ckpt "clip-E5-v6-epoch=06-val_accuracy=0.98.ckpt"

srun python main_whatsup_eval.py --model "clip" \
   --ckpt "clip-E6-v7-epoch=06-val_accuracy=0.99.ckpt"

srun python main_whatsup_eval.py --model "clip" \
   --ckpt "clip-E6-v8-epoch=25-val_accuracy=1.00.ckpt"

srun python main_whatsup_eval.py --model "clip" \
   --ckpt "clip-E6-v9-epoch=06-val_accuracy=1.00.ckpt"
