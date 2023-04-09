#!/bin/bash
#SBATCH --job-name=contextual
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=75G               # memory (per node)
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
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=${data} \
										--lambas=.001 \
										--optimizer=momentum \
										--lr=0.1 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--dptype True \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_ARMWideResNet_${method}" \
										--max_epoch 200 \
										--seed ${seed} \
										--y_noise ${y_noise} \
