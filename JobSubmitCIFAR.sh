#!/bin/bash
#SBATCH --job-name=dropout_train_%j
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=85GB               # memory (per node)
#SBATCH --time=2-5:50            # time (DD-HH:MM)
#SBATCH --error=/home/mila/c/chris.emezue/GFNDropout/slurmerror_gfn_dropout-%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/GFNDropout/slurmoutput_gfn_dropout-%j.txt

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
#conda activate GFlownets

source /home/mila/c/chris.emezue/gsl-env/bin/activate



###pretraining
#python Pretrain_MNIST.py --name "MNIST_Suprise"

data=$1

method=$2

dim=$3

p=$4

RewardType=$5

DataRatio=$6

seed=$7

FOLDER='Results_Final_18_600_2_CIFAR_Final'


python Run_training_CIFAR.py --Data $data --Method $method --Hidden_dim $dim --p $p --seed $seed --DataRatio $DataRatio --Epochs 800 --RewardType ${RewardType} --folder $FOLDER 