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


srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 2 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name clip_finetuning_batchsize2 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v1

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 2 --max_steps 10000 --accumulate_grad_batches 4 \
   --run_name clip_ft_batch_2 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v1

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 2 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name clip_ft_batch_2 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v3

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 2 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name clip_ft_batch_2 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v4

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 2 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name clip_ft_batch_2 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 2 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name clip_ft_batch_2 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 2 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name clip_ft_batch_2 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v7

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 2 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name clip_ft_batch_2 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 4 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name clip_ft_batch_4 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 2 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name clip_v8 --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8