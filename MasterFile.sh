#!/bin/bash


# ############GFN based models
#declare -a all_data=("MNIST" "CIFAR10" "SVHN")
declare -a all_data=("CIFAR10")

#declare -a all_data=("MNIST")

#declare -a all_data=("CIFAR10")

#declare -a all_methods=("RESNET_GFFN" "RESNET_nodropout" "RESNET_StandoutAll" "RESNET_dropoutAll" "RESNET_SVDAll")
#declare -a all_methods=("RESNET_GFFN")
declare -a all_methods=("RESNET_GFFN")

#declare -a all_methods=("RESNET_GFFN" "RESNET_nodropout" "RESNET_StandoutAll" "RESNET_dropoutAll" "RESNET_SVDAll")

declare -a all_dim=(1024)

#declare -a all_p=(0.1 0.2 0.5 0.7 0.9)
declare -a all_p=(0.2)


declare -a RewardTypes=(2)
#declare -a RewardTypes=(0)


declare -a All_DataRatio=(0.1)


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

							bash JobSubmit.sh $data $method $dim $p $RewardType $DataRatio $round	
						done
					done
				done
			done
		done
	done
done
