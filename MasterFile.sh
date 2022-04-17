#!/bin/bash



# ############GFN based models
#declare -a all_data=("MNIST" "CIFAR10")
declare -a all_data=("SVHN")

declare -a all_methods=("MLP_GFNDB"  "MLP_GFNFM")


declare -a all_dim=(50 100 200)
#declare -a all_dim=(10)

declare -a all_p=(0.2 0.5 0.8)
#declare -a all_p=(0.2)

declare -a OODRewards=(1 0)


declare -a all_rounds=(1)


for data in "${all_data[@]}"
do

	for method in "${all_methods[@]}"
	do

		for dim in "${all_dim[@]}"
		do

			for p in "${all_p[@]}"
			do
				for OODReward in "${OODRewards[@]}"
				do
					for round in "${all_rounds[@]}"
					do

						sbatch JobSubmit.sh $data $method $dim $p $OODReward $round	
					done
				done
			done
		done
	done
done



# ###no dropout and SVD ( cannot adjust p)
#declare -a all_data=("MNIST" "CIFAR10")
declare -a all_data=( "SVHN")
declare -a all_methods=("MLP_nodropout" "MLP_SVD")
#declare -a all_methods=("MLP_GFNDB" "MLP" "MLP_SVD" "MLP_Standout")
#declare -a all_methods=( "MLP_GFNFM" )

declare -a all_dim=(50 100 200)
#declare -a all_dim=(10)

declare -a all_p=(0)
#declare -a all_p=(0.2)

declare -a OODRewards=(0)

#declare -a all_rounds=(1 2 3)
declare -a all_rounds=(1)



for data in "${all_data[@]}"
do

	for method in "${all_methods[@]}"
	do

		for dim in "${all_dim[@]}"
		do

			for p in "${all_p[@]}"
			do
				for OODReward in "${OODRewards[@]}"
				do
					for round in "${all_rounds[@]}"
					do

						sbatch JobSubmit.sh $data $method $dim $p $OODReward $round	
					done
				done
			done
		done
	done
done

# #####standout and standard drouput
#declare -a all_data=("MNIST" "CIFAR10")
declare -a all_data=( "SVHN")
declare -a all_methods=("MLP_dropout" "MLP_Standout")
#declare -a all_methods=("MLP_GFNDB" "MLP" "MLP_SVD" "MLP_Standout")
#declare -a all_methods=( "MLP_GFNFM" )

declare -a all_dim=(50 100 200)
#declare -a all_dim=(10)

declare -a all_p=(0.2 0.5 0.8)
#declare -a all_p=(0.2)

declare -a OODRewards=(0)

#declare -a all_rounds=(1 2 3)
declare -a all_rounds=(1)


for data in "${all_data[@]}"
do

	for method in "${all_methods[@]}"
	do

		for dim in "${all_dim[@]}"
		do

			for p in "${all_p[@]}"
			do
				for OODReward in "${OODRewards[@]}"
				do
					for round in "${all_rounds[@]}"
					do

						sbatch JobSubmit.sh $data $method $dim $p $OODReward $round	
					done
				done
			done
		done
	done
done
