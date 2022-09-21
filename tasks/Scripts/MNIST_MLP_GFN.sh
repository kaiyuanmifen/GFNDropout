#!/bin/bash
#SBATCH --job-name=MLP
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-2:59            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets



###pretraining
#python Pretrain_MNIST.py --name "MNIST_Suprise"


python -u ../image_classification/main.py train \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "none" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--beta 1 \
										--max_epoch 200 \
										--mlp_dr 0.9 \
										#--start_model "../../checkpoints/MLP_GFN_MNIST_MLP_GFN_both_NN_base" \
										

