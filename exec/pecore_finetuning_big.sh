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


srun python main.py --model "pecore" \
   --lr 1e-6 --batch_size 200 --max_steps 5000 --accumulate_grad_batches 1 \
   --run_name proba_E0 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v3

srun python main.py --model "pecore" \
   --lr 1e-6 --batch_size 200 --max_steps 5000 --accumulate_grad_batches 1 \
   --run_name E0 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v4

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 200 --max_steps 5000 --accumulate_grad_batches 1 \
#    --run_name E0 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v3

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 200 --max_steps 5000 --accumulate_grad_batches 1 \
#    --run_name E0 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v4

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 200 --max_steps 5000 --accumulate_grad_batches 1 \
#    --run_name E0 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 200 --max_steps 5000 --accumulate_grad_batches 1 \
#    --run_name E0 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 200 --max_steps 5000 --accumulate_grad_batches 1 \
#    --run_name E0 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v7

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 200 --max_steps 5000 --accumulate_grad_batches 1 \
#    --run_name E0 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8








# BATCH 128
# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 128 --max_steps 1000 --accumulate_grad_batches 1 \
#    --run_name E1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v1

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 128 --max_steps 1000 --accumulate_grad_batches 1 \
#    --run_name E1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v2

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 128 --max_steps 1000 --accumulate_grad_batches 1 \
#    --run_name E1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v3

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 128 --max_steps 1000 --accumulate_grad_batches 1 \
#    --run_name E1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v4

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 128 --max_steps 1000 --accumulate_grad_batches 1 \
#    --run_name E1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 128 --max_steps 1000 --accumulate_grad_batches 1 \
#    --run_name E1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 128 --max_steps 1000 --accumulate_grad_batches 1 \
#    --run_name E1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v7

# srun python main.py --model "pecore" \
#    --lr 1e-6 --batch_size 128 --max_steps 1000 --accumulate_grad_batches 1 \
#    --run_name E1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8