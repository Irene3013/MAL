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


# ID
srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 1 \
   --run_name ID_P_b1 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 2 \
   --run_name ID_P_b1 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 3 \
   --run_name ID_P_b1 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 4 \
   --run_name ID_P_b1 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

# OOD
srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 1 \
   --run_name OOD_P_b1 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 2 \
   --run_name OOD_P_b1 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 3 \
   --run_name OOD_P_b1 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 4 \
   --run_name OOD_P_b1 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9


# EVALS##################

# ID
srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 1 \
   --run_name ID_P_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 2 \
   --run_name ID_P_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 3 \
   --run_name ID_P_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 4 \
   --run_name ID_P_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

# OOD
srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 1 \
   --run_name OOD_P_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 2 \
   --run_name OOD_P_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 3 \
   --run_name OOD_P_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 5000 --test_paraphrase 4 \
   --run_name OOD_P_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9