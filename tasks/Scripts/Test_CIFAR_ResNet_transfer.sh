#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=45G               # memory (per node)
#SBATCH --time=0-0:50            # time (DD-HH:MM)


###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets


dataset=$1

method=$2

subset_size=$3

seed=$4







declare -a all_subsetsize=(512 1024 2048 4096 8192)



for subsetsize in "${all_subsetsize[@]}"
do

	model_file="../../../saved_checkpoint/transfer_${dataset}/transfer_${dataset}_${method}_${subset_size}_${seed}.model"

	if [ $method == "Contextual" ]
	then
		echo "using ${method}"
		python ../image_classification/main.py test \
												--model=ResNet_Con \
												--GFN_dropout False \
												--dataset=${dataset} \
												--lambas=.001 \
												--schedule_milestone="[60, 120]" \
												--add_noisedata=False \
												--dptype True \
												--fixdistrdp False \
												--ctype "Bernoulli" \
												--dropout_distribution 'bernoulli' \
												--model_name "_ARMWideResNet_Contextual" \
		 										--load_file=${model_file}   \




	elif [ $method == "Concrete" ]
	then

		echo "using ${method}"
		python ../image_classification/main.py test \
												--model=ResNet_Con \
												--GFN_dropout False \
												--dataset=${dataset} \
												--add_noisedata=False \
												--concretedp True \
												--dptype False \
												--fixdistrdp False \
												--ctype "Bernoulli" \
												--dropout_distribution 'bernoulli' \
												--model_name "_ARMWideResNet_Concrete" \
		 										--load_file=${model_file}  \

 	else
		
		echo "using ${method}"

		python ../image_classification/main.py test \
												--model=ResNet_GFN \
												--GFN_dropout True \
												--dropout_rate 0.5 \
												--dataset=${dataset} \
												--lambas=.001 \
												--add_noisedata=False \
												--concretedp False \
												--dptype False \
												--fixdistrdp False \
												--ctype "Bernoulli" \
												--dropout_distribution 'bernoulli' \
												--model_name "_ResNet_GFN" \
												--mask ${method} \
												--BNN False \
		 										--load_file=${model_file} \


	fi
done



