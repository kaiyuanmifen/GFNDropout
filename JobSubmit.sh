#!/bin/bash
#SBATCH --job-name=dropout_train
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-5:50            # time (DD-HH:MM)
#SBATCH --error=/home/mila/c/chris.emezue/GFNDropout/slurmerror_gfn.txt
#SBATCH --output=/home/mila/c/chris.emezue/GFNDropout/slurmoutput_gfn.txt

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

OODReward=$5

seed=$6


python Run_training.py --Data $data --Method $method --Hidden_dim $dim --p $p --seed $seed --Epochs 200 --OODReward ${OODReward}
