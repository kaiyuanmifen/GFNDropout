#!/bin/bash


# ############GFN based models
declare -a all_data=("CIFAR10")
#declare -a all_data=("CIFAR10" )

declare -a all_methods=("MLP_GFFN")


declare -a all_dim=(64)

#declare -a all_p=(0.1 0.2 0.5 0.7 0.9)
declare -a all_p=(0.5)

#declare -a OODRewards=(1 0)
declare -a RewardTypes=(2)

declare -a All_DataRatio=(0.05)


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

							./JobSubmit.sh $data $method $dim $p $RewardTypes $DataRatio $round	
						done
					done
				done
			done
		done
	done
done





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


