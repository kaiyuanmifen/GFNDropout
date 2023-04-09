#!/bin/bash
#SBATCH --job-name=MLP
#SBATCH --gres=gpu:rtx8000:1               # Number of GPUs (per node)
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


data=$1

method=$2

seed=$3



python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=${data} \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask ${method} \
										--BNN False \
										--model_name "_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 200 \
										--seed ${seed} \
										



python -u ../image_classification/main.py test_rotate \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=${data} \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask ${method} \
										--BNN False \
										--model_name "_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 200 \
										--seed ${seed} \
