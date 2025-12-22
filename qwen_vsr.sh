#!/bin/bash

#SBATCH --job-name=qwen2_vsr                 # Name of the process
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

# srun python main.py --model "qwen2" \
#    --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
#    --run_name vsr_qwen_random --evaluate --dataset vsr \
#    --root /gaueko0/users/ietxarri010/MAL/data --precision 32 --variant zeroshot

srun python main.py --model "qwen2" \
   --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
   --run_name whatsup_qwen_images --evaluate --dataset whatsup \
   --root /gaueko0/users/ietxarri010/MAL/data --precision 32 --variant images

srun python main.py --model "qwen2" \
   --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
   --run_name whatsup_qwen_clevr --evaluate --dataset whatsup \
   --root /gaueko0/users/ietxarri010/MAL/data --precision 32 --variant clevr

srun python main.py --model "qwen2" \
   --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
   --image_path /gaueko0/users/ietxarri010/MAL/data/raw/COCO_spatial/val2017 \
   --run_name coco_qwen_one --evaluate --dataset cocospatial \
   --root /gaueko0/users/ietxarri010/MAL/data --precision 32 --variant one

srun python main.py --model "qwen2" \
   --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
   --image_path /gaueko0/users/ietxarri010/MAL/data/raw/COCO_spatial/val2017 \
   --run_name coco_qwen_two --evaluate --dataset cocospatial \
   --root /gaueko0/users/ietxarri010/MAL/data --precision 32 --variant two

srun python main.py --model "qwen2" \
   --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
   --image_path /gaueko0/users/ietxarri010/MAL/data/raw/GQA_spatial/vg_images \
   --run_name gqa_qwen_one --evaluate --dataset gqaspatial \
   --root /gaueko0/users/ietxarri010/MAL/data --precision 32 --variant one

srun python main.py --model "qwen2" \
   --lr 2e-5 --batch_size 1 --max_steps 20000 --accumulate_grad_batches 2 \
   --image_path /gaueko0/users/ietxarri010/MAL/data/raw/GQA_spatial/vg_images \
   --run_name gqa_qwen_two --evaluate --dataset gqaspatial \
   --root /gaueko0/users/ietxarri010/MAL/data --precision 32 --variant two
