#!/bin/bash
#SBATCH --job-name=GFN
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-6:50            # time (DD-HH:MM)


###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets




 CUDA_LAUNCH_BLOCKING=1 python ../image_classification/main.py train \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dataset=cifar100 \
										--lambas=.001 \
										--optimizer=momentum \
										--lr=0.1 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "upNdown" \
										--BNN False \
										--max_epoch 200 \
										--mlp_dr 0.9 \
										--Tune_last_layer_only True \
										--use_pretrained True \	