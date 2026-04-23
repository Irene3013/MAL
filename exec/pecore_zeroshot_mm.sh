#!/bin/bash

#SBATCH --job-name=pecore_zs                     # Name of the process
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
#    --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
#    --run_name pecore_v1 --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v1

# srun python main.py --model "pecore" \
#    --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
#    --run_name pecore_v2 --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v2

# srun python main.py --model "pecore" \
#    --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
#    --run_name pecore_v3 --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v3

# srun python main.py --model "pecore" \
#    --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
#    --run_name pecore_v4 --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v4

srun python main.py --model "pecore" \
   --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
   --run_name pecore_v7 --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v7

srun python main.py --model "pecore" \
   --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
   --run_name pecore_v8 --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8