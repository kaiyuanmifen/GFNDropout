#!/bin/bash

# ############GFN based models
declare -a all_data=("MNIST")
#declare -a all_data=("CIFAR10" )

#declare -a all_methods=("CNN_GFNDB" "MLP_GFNDB" "CNN_GFNFM"  "MLP_GFNFM")

declare -a all_methods=("MLP_GFFN" "MLP_dropoutAll" "MLP_StandoutAll" "MLP_nodropout" "MLP_SVDAll")

#declare -a all_methods=("MLP_GFFN")


#declare -a all_dim=(20 40 80)
declare -a all_dim=(1024)

#declare -a all_p=(0.1 0.2 0.5 0.7 0.9)
declare -a all_p=(0.5)

#declare -a OODRewards=(1 0)
declare -a OODRewards=(0)

declare -a All_DataRatio=(0.05 0.1 1.0)


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
					for OODReward in "${OODRewards[@]}"
					do
						for round in "${all_rounds[@]}"
						do

							./JobSubmit.sh $data $method $dim $p $OODReward $DataRatio $round	
						done
					done
				done
			done
		done
	done
done




# # ############GFN based models
# declare -a all_data=("MNIST" "CIFAR10")
# #declare -a all_data=("CIFAR10" )

# #declare -a all_methods=("CNN_GFNDB" "MLP_GFNDB" "CNN_GFNFM"  "MLP_GFNFM")

# declare -a all_methods=("MLP_GFFN"  "MLP_dropoutAll" "MLP_nodropout" "MLP_StandoutAll" "MLP_SVDAll")

# #declare -a all_methods=("MLP_GFFN")


# #declare -a all_dim=(20 40 80)
# declare -a all_dim=(256)

# #declare -a all_p=(0.1 0.2 0.5 0.7 0.9)
# declare -a all_p=(0.5)

# #declare -a OODRewards=(1 0)
# declare -a OODRewards=(0)

# declare -a All_DataRatio=(0.05 0.1 0.2 0.5 1.0)


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
# 					for OODReward in "${OODRewards[@]}"
# 					do
# 						for round in "${all_rounds[@]}"
# 						do

# 							./JobSubmit.sh $data $method $dim $p $OODReward $DataRatio $round	
# 						done
# 					done
# 				done
# 			done
# 		done
# 	done
# done



# ###no dropout and SVD ( cannot adjust p)
# declare -a all_data=("MNIST" "CIFAR10" "SVHN")
# #declare -a all_data=( "SVHN")
# declare -a all_methods=("CNN_SVD" "CNN_nodropout")
# #declare -a all_methods=("MLP_GFNDB" "MLP" "MLP_SVD" "MLP_Standout")
# #declare -a all_methods=( "MLP_GFNFM" )

# declare -a all_dim=(20 40 80)
# #declare -a all_dim=(10)

# declare -a all_p=(0)
# #declare -a all_p=(0.2)

# declare -a OODRewards=(0)

# #declare -a all_rounds=(1 2 3)
# declare -a all_rounds=(1)



# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do

# 		for dim in "${all_dim[@]}"
# 		do

# 			for p in "${all_p[@]}"
# 			do
# 				for OODReward in "${OODRewards[@]}"
# 				do
# 					for round in "${all_rounds[@]}"
# 					do

# 						sbatch JobSubmit.sh $data $method $dim $p $OODReward $round	
# 					done
# 				done
# 			done
# 		done
# 	done
# done

# # # #####standout and standard drouput
# declare -a all_data=("MNIST" "CIFAR10" "SVHN")
# #declare -a all_data=( "SVHN")
# #declare -a all_methods=("MLP_dropout" "MLP_Standout")
# declare -a all_methods=("CNN_Standout" "CNN_dropout")

# #declare -a all_methods=("MLP_GFNDB" "MLP" "MLP_SVD" "MLP_Standout")
# #declare -a all_methods=( "MLP_GFNFM" )

# declare -a all_dim=(20 40 80)
# #declare -a all_dim=(10)

# declare -a all_p=(0.1 0.2 0.5 0.7 0.9)
# #declare -a all_p=(0.2)

# declare -a OODRewards=(0)

# #declare -a all_rounds=(1 2 3)
# declare -a all_rounds=(1)


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do

# 		for dim in "${all_dim[@]}"
# 		do

# 			for p in "${all_p[@]}"
# 			do
# 				for OODReward in "${OODRewards[@]}"
# 				do
# 					for round in "${all_rounds[@]}"
# 					do

# 						sbatch JobSubmit.sh $data $method $dim $p $OODReward $round	
# 					done
# 				done
# 			done
# 		done
# 	done
# done
