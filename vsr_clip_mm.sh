#!/bin/bash

#SBATCH --job-name=vsr_zs_clip                 # Name of the process
#SBATCH --cpus-per-task=2                      # Number of CPU cores (2 is reasonable)
#SBATCH --gres=gpu:1                           # Number of GPUs (usually light processes only need 1)
#SBATCH --mem=64G                              # RAM memory needed (8-16GB is reasonable for our servers, sometimes you'll need more)
#SBATCH --output=/gaueko0/users/ietxarri010/MAL/log_prompt.log
#SBATCH --error=/gaueko0/users/ietxarri010/MAL/error_prompt.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=irene.etxarri@ehu.eus

#source /gscratch/users/asalaberria009/env/p39-cu115/bin/activate
source /gaueko0/users/ietxarri010/env/nire_env/bin/activate

#export TRANSFORMERS_CACHE="/ncache/hub/"
srun python main.py --model "clip" --target_model ViT-B/32  \
   --lr 2e-5 --batch_size 4 --max_steps 20000 --accumulate_grad_batches 2 \
   --run_name vsr_zs_clip_finetuning --train --evaluate --dataset vsr \
   --root /gaueko0/users/ietxarri010/MAL --precision 32 --variant zeroshot