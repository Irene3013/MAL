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

# BASELINE
srun python main_whatsup_eval.py --model "pecore" --output_name ZS  

# ID
srun python main_whatsup_eval.py --model "pecore" --output_name ID --ckpt pecore-ID_b2-v5-epoch=02-val_accuracy=1.00.ckpt 
   
# OOD
srun python main_whatsup_eval.py --model "pecore" --output_name OOD --ckpt pecore-OOD_b2-v8-epoch=05-val_accuracy=1.00.ckpt

# ID_P
srun python main_whatsup_eval.py --model "pecore" --output_name ID_P1 --ckpt pecore-ID_P_b1-v6-epoch=00-val_accuracy=1.00.ckpt

srun python main_whatsup_eval.py --model "pecore" --output_name ID_P2 --ckpt pecore-ID_P_b1-v6-epoch=00-val_accuracy=1.00-v1.ckpt

srun python main_whatsup_eval.py --model "pecore" --output_name ID_P3 --ckpt pecore-ID_P_b1-v6-epoch=00-val_accuracy=1.00-v2.ckpt 

srun python main_whatsup_eval.py --model "pecore" --output_name ID_P4 --ckpt pecore-ID_P_b1-v6-epoch=00-val_accuracy=1.00-v3.ckpt 

# OOD_P
srun python main_whatsup_eval.py --model "pecore" --output_name OOD_P1 --ckpt pecore-OOD_P_b1-v9-epoch=00-val_accuracy=1.00.ckpt  

srun python main_whatsup_eval.py --model "pecore" --output_name OOD_P2 --ckpt pecore-OOD_P_b1-v9-epoch=01-val_accuracy=0.69.ckpt 

srun python main_whatsup_eval.py --model "pecore" --output_name OOD_P3 --ckpt pecore-OOD_P_b1-v9-epoch=01-val_accuracy=1.00.ckpt 

srun python main_whatsup_eval.py --model "pecore" --output_name OOD_P4 --ckpt pecore-OOD_P_b1-v9-epoch=01-val_accuracy=1.00-v1.ckpt 