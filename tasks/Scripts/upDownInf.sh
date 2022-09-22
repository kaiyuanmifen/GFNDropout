#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-2:50            # time (DD-HH:MM)


###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1

source /home/mila/c/chris.emezue/gsl-env/bin/activate


#upNdown



python -m pdb ../image_classification/main.py test \
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
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="/home/mila/c/chris.emezue/GFNDropout/checkpoints/upNdown199.model" 



