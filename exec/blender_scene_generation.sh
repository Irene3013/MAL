#!/bin/bash

#SBATCH --job-name=blender_scenes              # Name of the process
#SBATCH --cpus-per-task=2                      # Number of CPU cores (2 is reasonable)
#SBATCH --gres=gpu:1                           # Number of GPUs (usually light processes only need 1)
#SBATCH --mem=64G                              # RAM memory needed (8-16GB is reasonable for our servers, sometimes you'll need more)
#SBATCH --output=/gaueko0/users/ietxarri010/MAL/log_prompt.log
#SBATCH --error=/gaueko0/users/ietxarri010/MAL/error_prompt.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=irene.etxarri@ehu.eus

source /gaueko0/users/ietxarri010/env/bpy_env/bin/activate

#export TRANSFORMERS_CACHE="/ncache/hub/"

srun python /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset/create_blender_scenes.py \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --version v1 


srun python /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset/create_blender_scenes.py \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --version v2 


srun python /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset/create_blender_scenes.py \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --version v3 


srun python /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset/create_blender_scenes.py \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --version v4 


srun python /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset/create_blender_scenes.py \
   --root /gaueko0/users/ietxarri010/MAL/data/raw/RelationsDataset --version v5 
