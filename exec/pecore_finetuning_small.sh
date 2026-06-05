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

# ID
srun python main.py --model "pecore" \
   --lr 1e-6 --batch_size 32 --max_steps 5000 \
   --run_name Ps_ID --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

srun python main.py --model "pecore" \
   --lr 1e-6 --batch_size 1 --max_steps 5000  \
   --run_name Ps_ID --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

# OOD
srun python main.py --model "pecore" \
   --lr 1e-6 --batch_size 32 --max_steps 5000 \
   --run_name Ps_OOD --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "pecore" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 \
   --run_name Ps_OOD --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8