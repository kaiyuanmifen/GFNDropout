#!/bin/bash
#SBATCH --job-name=GFN
#SBATCH --gres=gpu:rtx8000:1               # Number of GPUs (per node)
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-7:50            # time (DD-HH:MM)


###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets


data=$1

method=$2

y_noise=$3

seed=$4


python ../image_classification/main.py train \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dataset=${data} \
										--lambas=.001 \
										--optimizer=momentum \
										--lr=0.05 \
										--schedule_milestone="[25, 40]" \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_ResNet_GFN" \
										--mask ${method} \
										--BNN False \
										--max_epoch 200 \
										--seed ${seed} \
										--y_noise ${y_noise} \
										