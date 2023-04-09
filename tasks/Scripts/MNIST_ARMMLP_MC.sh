#!/bin/bash
#SBATCH --job-name=contextual
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-3:50            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets



###pretraining
#python Pretrain_MNIST.py --name "MNIST_Suprise"

python ../image_classification/main.py train \
										--model=ARMMLP \
										--GFN_dropout False \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--fixdistrdp True \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_MNIST_ARMMLP_MC"
