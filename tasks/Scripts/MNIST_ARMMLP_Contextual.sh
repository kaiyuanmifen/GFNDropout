#!/bin/bash
#SBATCH --job-name=contextual
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
										--model=ARMMLP \
										--GFFN_dropout False \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype True \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_MNIST_ARMMLP_Contextual" \
										--max_epoch 200 \


# python ../image_classification/main.py train \
# 										--model=ARMMLP \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--model_name "_MNIST_ARMMLP_Contextual"