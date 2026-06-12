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

# ID - BATCH SIZE ####################################################################################################
srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 \
   --run_name ID_z --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 200 --max_steps 10000 \
   --run_name ID_b200 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5
   
srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 128 --max_steps 10000 \
   --run_name ID_b128 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 64 --max_steps 10000 \
   --run_name ID_b64 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 32 --max_steps 10000 \
   --run_name ID_b32 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 16 --max_steps 10000 \
   --run_name ID_b16 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 4 --max_steps 10000 \
   --run_name ID_b4 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 2 --max_steps 10000 \
   --run_name ID_b2 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 \
   --run_name ID_b1 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v5

# OOD - BATCH SIZE ####################################################################################################

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 \
   --run_name OOD_z --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 200 --max_steps 10000 \
   --run_name OOD_b200 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8
   
srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 128 --max_steps 10000 \
   --run_name OOD_b128 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 64 --max_steps 10000 \
   --run_name OOD_b64 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 32 --max_steps 10000 \
   --run_name OOD_b32 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 16 --max_steps 10000 \
   --run_name OOD_b16 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 4 --max_steps 10000 \
   --run_name OOD_b4 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 2 --max_steps 10000 \
   --run_name OOD_b2 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 \
   --run_name OOD_b1 --train --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v8

# PARAFRASES ####################################################################################################
# ID
# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 1 \
#    --run_name ID_P_b1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 2 \
#    --run_name ID_P_b1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 3 \
#    --run_name ID_P_b1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 4 \
#    --run_name ID_P_b1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

# # OOD
# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 1 \
#    --run_name OOD_P_b1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 2 \
#    --run_name OOD_P_b1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 3 \
#    --run_name OOD_P_b1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

# srun python main.py --model "clip" \
#    --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 4 \
#    --run_name OOD_P_b1 --train --evaluate --dataset rel \
#    --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9


# EVALS##################

# ID
srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 1 \
   --run_name ID_P1_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 2 \
   --run_name ID_P2_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 3 \
   --run_name ID_P3_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 4 \
   --run_name ID_P4_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v6

# OOD
srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 1 \
   --run_name OOD_P1_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 2 \
   --run_name OOD_P2_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 3 \
   --run_name OOD_P3_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9

srun python main.py --model "clip" \
   --lr 1e-6 --batch_size 1 --max_steps 10000 --test_paraphrase 4 \
   --run_name OOD_P4_eval --evaluate --dataset rel \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --precision 32 --variant v9