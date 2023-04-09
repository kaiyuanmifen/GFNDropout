#!/bin/bash
#SBATCH --job-name=MLP
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=45G               # memory (per node)
#SBATCH --time=0-1:59            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets



python ../image_classification/main.py train \
										--model=ARMWideResNet \
										--GFN_dropout False \
										--dataset=cifar10 \
										--lambas=.001 \
										--optimizer=momentum \
										--lr=0.1 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--concretedp True \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ARMWideResNet_Concrete" \
										--max_epoch 100 \


python ../image_classification/main.py train \
										--model=ARMWideResNet \
										--GFN_dropout False \
										--dataset=cifar10 \
										--lambas=.001 \
										--optimizer=momentum \
										--lr=0.1 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--dptype True \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ARMWideResNet_Contextual" \
										--max_epoch 100 \


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
										--mask "random" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 100 \
										#--start_model "../../checkpoints/MLP_GFN_MNIST_MLP_GFN_both_NN_base" \
										

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
										--beta 0.001 \
										--max_epoch 100 \



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
										--mask "topdown" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 100 \




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
										--mask "bottomup" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 100 \