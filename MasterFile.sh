#!/bin/bash


###train model for task


declare -a all_data=("MNIST" "CIFAR10")
#declare -a all_data=("MNIST")


declare -a all_methods=("MLP" "MLP_GFN" "MLP_SVD" "MLP_Standout")
#declare -a all_methods=("MLP_GFN")


declare -a all_dim=(10 20 30)
#declare -a all_dim=(10)

declare -a all_p=(0.2 0.5 0.8)
#declare -a all_p=(0.5)


declare -a all_rounds=(1 2 3)


for data in "${all_data[@]}"
do

	for method in "${all_methods[@]}"
	do

		for dim in "${all_dim[@]}"
		do

			for p in "${all_p[@]}"
			do

				for round in "${all_rounds[@]}"
				do

					sbatch JobSubmit.sh $data $method $dim $p $round	
				done
			done
		done
	done
done
