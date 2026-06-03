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



# ID
srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 200 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name ID --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 32 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name ID --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name ID --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

# OOD
srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 200 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name OOD --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 32 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name OOD --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 --accumulate_grad_batches 1 \
   --run_name OOD --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

# ID_P - OOD_P
# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 200 --max_steps 10000 --accumulate_grad_batches 1 \
#    --run_name W1 --evaluate --dataset rel --output_path /gaueko0/users/ietxarri010/out_p4/ \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6


# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 200 --max_steps 10000 --accumulate_grad_batches 1 \
#    --run_name W1 --evaluate --dataset rel --output_path /gaueko0/users/ietxarri010/out_p4/ \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9
