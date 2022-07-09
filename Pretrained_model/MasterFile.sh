#!/bin/bash

#SBATCH --job-name=dropout_train
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=95G               # memory (per node)
#SBATCH --time=0-5:50            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

# module load anaconda/3
# module load cuda/11.1
# #conda activate GNN
# conda activate GFlownets






###########pretrained model experiment


#declare -a AllData=("SVHN" )

#declare -a AllData=("CIFAR10" "CIFAR100" "SVHN")
declare -a AllData=("CIFAR10")

#declare -a AllDropout=("GFN" "Random" "None")

declare -a AllDropout=("GFN")

#declare -a AllHowToTrain=("LinearProbing")

declare -a AllHowToTrain=("LinearProbing" "FineTuning")


declare -a AllSeeds=(1 2 3) 

for Data in "${AllData[@]}"
do

	for Dropout in "${AllDropout[@]}"
	do


		for HowToTrain in "${AllHowToTrain[@]}"
		do

			for Seed in "${AllSeeds[@]}"
			do

					sbatch JobSubmit_pretrained.sh $Data $Dropout $HowToTrain $Seed
			done
		done

	done
done