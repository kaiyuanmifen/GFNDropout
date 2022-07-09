#!/bin/bash

#SBATCH --job-name=dropout_train
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=95G               # memory (per node)
#SBATCH --time=0-5:50            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets






# ############GFN based models
#declare -a all_data=("MNIST" "CIFAR10")
#declare -a all_data=("CIFAR10" "MNIST" )

declare -a all_data=("MNIST")

#declare -a all_methods=("MLP_GFFN" "MLP_nodropout" "MLP_StandoutAll" "MLP_dropoutAll" "MLP_SVDAll")
declare -a all_methods=("MLP_GFFN")
#declare -a all_methods=("RESNET_dropoutAll")

#declare -a all_methods=("MLP_GFFN" "MLP_nodropout" "MLP_StandoutAll" "MLP_dropoutAll" "MLP_SVDAll")

declare -a all_dim=(256)

#declare -a all_p=(0.1 0.2 0.5 0.7 0.9)
declare -a all_p=(0.5)


declare -a RewardTypes=(0)
#declare -a RewardTypes=(0)


declare -a All_DataRatio=(0.01)


declare -a all_rounds=(1) 


for data in "${all_data[@]}"
do

	for method in "${all_methods[@]}"
	do

		for dim in "${all_dim[@]}"
		do

			for p in "${all_p[@]}"
			do
				for DataRatio in "${All_DataRatio[@]}"
				do
					for RewardType in "${RewardTypes[@]}"
					do
						for round in "${all_rounds[@]}"
						do

							./JobSubmit.sh $data $method $dim $p $RewardType $DataRatio $round	
						done
					done
				done
			done
		done
	done
done






# # ############GFN based models
# #declare -a all_data=("MNIST" "CIFAR10")
# #declare -a all_data=("CIFAR10" "MNIST" )

# declare -a all_data=("CIFAR10")

# #declare -a all_methods=("MLP_GFFN" "MLP_nodropout" "MLP_StandoutAll" "MLP_dropoutAll" "MLP_SVDAll")
# declare -a all_methods=("RESNET_GFFN")
# #declare -a all_methods=("RESNET_dropoutAll")

# #declare -a all_methods=("MLP_GFFN" "MLP_nodropout" "MLP_StandoutAll" "MLP_dropoutAll" "MLP_SVDAll")

# declare -a all_dim=(256)

# #declare -a all_p=(0.1 0.2 0.5 0.7 0.9)
# declare -a all_p=(0.5)


# declare -a RewardTypes=(0 1 2)
# #declare -a RewardTypes=(0)


# declare -a All_DataRatio=(1)


# declare -a all_rounds=(1) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do

# 		for dim in "${all_dim[@]}"
# 		do

# 			for p in "${all_p[@]}"
# 			do
# 				for DataRatio in "${All_DataRatio[@]}"
# 				do
# 					for RewardType in "${RewardTypes[@]}"
# 					do
# 						for round in "${all_rounds[@]}"
# 						do

# 							sbatch JobSubmit.sh $data $method $dim $p $RewardType $DataRatio $round	
# 						done
# 					done
# 				done
# 			done
# 		done
# 	done
# done




# # ############GFN based models
# declare -a all_data=("MNIST")
# #declare -a all_data=("CIFAR10" "SVHN")

# #declare -a all_methods=("MLP_GFFN" "MLP_dropoutAll" "MLP_StandoutAll" "MLP_nodropout" "MLP_SVDAll")

# declare -a all_methods=("MLP_GFFN")


# #declare -a all_dim=(20 40 80)
# declare -a all_dim=(1024)

# #declare -a all_p=(0.1 0.2 0.5 0.7 0.9)
# declare -a all_p=(0.5)

# declare -a RewardTypes=(2)

# declare -a All_DataRatio=(0.05 0.1 1.0)


# declare -a all_rounds=(1) 

# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do

# 		for dim in "${all_dim[@]}"
# 		do

# 			for p in "${all_p[@]}"
# 			do
# 				for DataRatio in "${All_DataRatio[@]}"
# 				do
# 					for RewardType in "${RewardTypes[@]}"
# 					do
# 						for round in "${all_rounds[@]}"
# 						do

# 							./JobSubmit.sh $data $method $dim $p $RewardTypes $DataRatio $round	
# 						done
# 					done
# 				done
# 			done
# 		done
# 	done
# done




############Gater name experiement


# #declare -a AllMethods=("GFFN" "GaterNet" "StandardDropout")
# declare -a AllMethods=("GFFN")

# #declare -a RewardTypes=(0)


# declare -a AllDepths=(20)


# declare -a AllData=("MNIST") 


# for method in "${AllMethods[@]}"
# do

# 	for depth in "${AllDepths[@]}"
# 	do

# 		for data in "${AllData[@]}"
# 		do

# 			./JobSubmit_gater.sh $data $method $depth
			
# 		done
# 	done
# done







# ###########pretrained model experiment


# declare -a AllDropout=("GFN" "Random" "None")

# #declare -a AllDropout=("GFN")


# declare -a AllSeeds=(1 2 3) 


# for Dropout in "${AllDropout[@]}"
# do

# 	for Seed in "${AllSeeds[@]}"
# 	do

# 			sbatch JobSubmit_pretrained.sh $Dropout $Seed
# 	done
# done