#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-2:50            # time (DD-HH:MM)


###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets





# python ../image_classification/main.py train \
# 										--model=ResNet_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.5 \
# 										--dataset=cifar100 \
# 										--lambas=.001 \
# 										--add_noisedata=False \
# 										--concretedp False \
# 										--dptype False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--model_name "_CIFAR_ResNet_GFN" \
# 										--mask "bottomup" \
# 										--BNN False \
# 										--augment_test=False \
# 										--subset_size=1024 \
										



# python ../image_classification/main.py test \
# 										--model=ARMWideResNet \
# 										--GFN_dropout False \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--optimizer=momentum \
# 										--lr=0.1 \
# 										--schedule_milestone="[60, 120]" \
# 										--add_noisedata=False \
# 										--dptype True \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
#  										--augment_test=False \
#  										--corruption_name="impulse_noise" \
#  										--corruption_severity=3 \
										

python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_ResNet_GFN" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=False \