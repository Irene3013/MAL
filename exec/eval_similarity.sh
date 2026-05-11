#!/bin/bash

#SBATCH --job-name=similarity                     # Name of the process
#SBATCH --cpus-per-task=2                      # Number of CPU cores (2 is reasonable)
#SBATCH --gres=gpu:1                           # Number of GPUs (usually light processes only need 1)
#SBATCH --mem=64G                              # RAM memory needed (8-16GB is reasonable for our servers, sometimes you'll need more)
#SBATCH --output=/gaueko0/users/ietxarri010/MAL/log_prompt.log
#SBATCH --error=/gaueko0/users/ietxarri010/MAL/error_prompt.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=irene.etxarri@ehu.eus

source /gaueko0/users/ietxarri010/env/nire_env/bin/activate

#export TRANSFORMERS_CACHE="/ncache/hub/"
# E5 - 4
# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 4 --max_steps 10000 --accumulate_grad_batches 1 \
#    --run_name E5 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v1

srun main_similarity.py --model "clip" --dataset rel --root '/gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset'  --variant v3 --output_path "/gaueko0/users/ietxarri010/MAL/sim/"
srun main_similarity.py --model "clip" --dataset rel --root '/gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset'  --variant v4 --output_path "/gaueko0/users/ietxarri010/MAL/sim/"
srun main_similarity.py --model "clip" --dataset rel --root '/gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset'  --variant v5 --output_path "/gaueko0/users/ietxarri010/MAL/sim/"
srun main_similarity.py --model "clip" --dataset rel --root '/gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset'  --variant v6 --output_path "/gaueko0/users/ietxarri010/MAL/sim/"

srun main_similarity.py --model "clip" --dataset rel --root '/gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset'  --variant v3 --paraphrase v4 --output_path "/gaueko0/users/ietxarri010/MAL/sim/"
srun main_similarity.py --model "clip" --dataset rel --root '/gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset'  --variant v5 --paraphrase v6 --output_path "/gaueko0/users/ietxarri010/MAL/sim/"