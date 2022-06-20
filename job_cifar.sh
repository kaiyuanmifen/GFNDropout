#!/bin/bash
#SBATCH --job-name=dropout_train_cifar
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=85GB               # memory (per node)
#SBATCH --time=2-5:50            # time (DD-HH:MM)
#SBATCH --error=/home/mila/c/chris.emezue/GFNDropout/slurmerror_gfn_dropout_cifar.txt
#SBATCH --output=/home/mila/c/chris.emezue/GFNDropout/slurmoutput_gfn_dropout_cifar.txt

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
#conda activate GFlownets

source /home/mila/c/chris.emezue/gsl-env/bin/activate


python -u train_cifar.py --epochs 500