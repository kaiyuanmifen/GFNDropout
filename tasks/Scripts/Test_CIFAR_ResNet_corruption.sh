#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:rtx8000:1               # Number of GPUs (per node)
#SBATCH --mem=55G               # memory (per node)
#SBATCH --time=0-1:50            # time (DD-HH:MM)


###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets


dataset=$1

method=$2

seed=$3


if [[ $dataset == "cifar10c" ]]
then
	dataset_trained="cifar10"
fi 

if [[ $dataset == "cifar100c" ]]
then
	dataset_trained="cifar100"
fi 


model_file="../../checkpoints/${dataset_trained}/${dataset_trained}_${method}_${seed}.model"


declare -a corruptions=("gaussian_noise" "snow" "frost")

declare -a severities=(1 2 3 4 5) 


for corruption in "${corruptions[@]}"
do

	for severity in "${severities[@]}"
	do

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
													--augment_test=False \
			 										--load_file=${model_file}   \
			 										--corruption_name=${corruption} \
		 											--corruption_severity=${severity} \



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
													--augment_test=False \
			 										--load_file=${model_file}  \
			 										--corruption_name=${corruption} \
		 											--corruption_severity=${severity} \

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
													--augment_test=False \
			 										--corruption_name=${corruption} \
		 											--corruption_severity=${severity} \
													--load_file=${model_file} \
													
		fi

	done 
done



