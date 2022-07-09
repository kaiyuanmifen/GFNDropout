#!/bin/bash
#SBATCH --job-name=dropout_gater
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=75G               # memory (per node)
#SBATCH --time=0-7:50            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets



###pretraining
#python Pretrain_MNIST.py --name "MNIST_Suprise"

data=$1

method=$2

dim=$3

p=$4

RewardType=$5

DataRatio=$6

seed=$7


python Run_training.py --Data $data --Method $method --Hidden_dim $dim --p $p --seed $seed --DataRatio $DataRatio --Epochs 400 --RewardType ${RewardType}
