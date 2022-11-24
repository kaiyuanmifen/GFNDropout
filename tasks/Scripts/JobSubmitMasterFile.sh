#!/bin/bash
#SBATCH --job-name=master
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=25G               # memory (per node)
#SBATCH --time=0-0:50            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets





###########training###########
#### resenet18 experiments


###GF
# ###GFN 
# declare -a all_data=("cifar100" "cifar10" )

# declare -a all_methods=("topdown" "bottomup" "none" "random")


# declare -a all_seeds=(1 2 3 4 5) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do


# 		for seed in "${all_seeds[@]}"
# 		do

# 			sbatch CIFAR_ResNet_GFN.sh $data $method $seed
# 		done
# 	done
# done

# ##oontextual 
# declare -a all_data=("cifar100" "cifar10" )

# declare -a all_methods=("Contextual")


# declare -a all_seeds=(1 2 3 4 5) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do


# 		for seed in "${all_seeds[@]}"
# 		do

# 			sbatch CIFAR_ResNet_Contextual.sh $data $method $seed
# 		done
# 	done
# done

# ##concrete 
# declare -a all_data=("cifar100" "cifar10" )

# declare -a all_methods=("Concrete")


# declare -a all_seeds=(1 2 3 4 5) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do


# 		for seed in "${all_seeds[@]}"
# 		do

# 			sbatch CIFAR_ResNet_Concrete.sh $data $method $seed
# 		done
# 	done
# done





# ####MLP MNIST


# ###GFN 
# declare -a all_data=("mnist")

# declare -a all_methods=("topdown")


# declare -a all_seeds=(1 2 3 4 5) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do


# 		for seed in "${all_seeds[@]}"
# 		do

# 			sbatch MNIST_MLP_GFN.sh $data $method $seed
# 		done
# 	done
# done

# ###GFN 
# declare -a all_data=("mnist" )

# declare -a all_methods=("topdown" "bottomup" "none" "random")


# declare -a all_seeds=(1 2 3 4 5) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do


# 		for seed in "${all_seeds[@]}"
# 		do

# 			sbatch MNIST_MLP_GFN.sh $data $method $seed
# 		done
# 	done
# done

# ##oontextual 
# declare -a all_data=("mnist" )

# declare -a all_methods=("Contextual")


# declare -a all_seeds=(1 2 3 4 5) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do


# 		for seed in "${all_seeds[@]}"
# 		do

# 			sbatch MNIST_ARMMLP_Contextual.sh $data $method $seed
# 		done
# 	done
# done

# ##concrete 
# declare -a all_data=("mnist" )

# declare -a all_methods=("Concrete")


# declare -a all_seeds=(1 2 3 4 5) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do


# 		for seed in "${all_seeds[@]}"
# 		do

# 			sbatch MNIST_ARMMLP_Concrete.sh $data $method $seed
# 		done
# 	done
# done




####train on data with augmented y
#### resenet18 experiments

# ###GFN 
# declare -a all_data=("cifar100" "cifar10" )

# declare -a all_methods=("topdown" "bottomup" "none" "random")


# declare -a all_seeds=(1) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do


# 		for seed in "${all_seeds[@]}"
# 		do

# 			sbatch CIFAR_ResNet_GFN.sh $data $method True $seed
# 		done
# 	done
# done

# ##oontextual 
# declare -a all_data=("cifar100" "cifar10" )

# declare -a all_methods=("Contextual")


# declare -a all_seeds=(1) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do


# 		for seed in "${all_seeds[@]}"
# 		do

# 			sbatch CIFAR_ResNet_Contextual.sh $data $method True $seed
# 		done
# 	done
# done

# ##concrete 
# declare -a all_data=("cifar100" "cifar10" )

# declare -a all_methods=("Concrete")


# declare -a all_seeds=(1) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do


# 		for seed in "${all_seeds[@]}"
# 		do

# 			sbatch CIFAR_ResNet_Concrete.sh $data $method True $seed
# 		done
# 	done
# done


####transfer to original cifar data

# ##GFN 
# declare -a all_data=("cifar100" "cifar10" )

# declare -a all_methods=("topdown" "bottomup" "none" "random")

# declare -a all_subsetsize=(512 1024 2048 4096 8192)

# declare -a all_seeds=(1) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do

# 		for subsetsize in "${all_subsetsize[@]}"
# 		do


# 			for seed in "${all_seeds[@]}"
# 			do

# 				 sbatch CIFAR_ResNet_GFN_transfer.sh $data $method $subsetsize $seed
# 			done

# 		done
# 	done
# done

# ##oontextual 
# declare -a all_data=("cifar100" "cifar10" )

# declare -a all_methods=("Contextual")

# declare -a all_subsetsize=(512 1024 2048 4096 8192)

# declare -a all_seeds=(1) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do
# 		for subsetsize in "${all_subsetsize[@]}"
# 		do

# 			for seed in "${all_seeds[@]}"
# 			do

# 				sbatch CIFAR_ResNet_Contextual_transfer.sh $data $method $subsetsize $seed
# 			done
# 		done
# 	done
# done

# ##concrete 
# declare -a all_data=("cifar100" "cifar10" )

# declare -a all_methods=("Concrete")

# declare -a all_subsetsize=(512 1024 2048 4096 8192)

# declare -a all_seeds=(1) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do

# 		for subsetsize in "${all_subsetsize[@]}"
# 		do


# 			for seed in "${all_seeds[@]}"
# 			do

# 				sbatch CIFAR_ResNet_Concrete_transfer.sh $data $method $subsetsize $seed
# 			done
# 		done
# 	done
# done









###### test on CIFAR10 and CIAFR100


# ###GFN 
declare -a all_data=("cifar10" "cifar100")

#declare -a all_methods=("bottomup" "topdown" "random" "Contextual" "Concrete")
declare -a all_methods=("topdown")


declare -a all_seeds=(1 2 3 4 5) 


for data in "${all_data[@]}"
do

	for method in "${all_methods[@]}"
	do


		for seed in "${all_seeds[@]}"
		do
			echo $data
			echo $method 
			echo $seed
			sbatch Test_CIFAR_ResNet.sh $data $method $seed
		done
	done
done




##### test on CIFAR10-c and CIAFR100-c


# ###GFN 
declare -a all_data=("cifar10c" "cifar100c")

#declare -a all_methods=("bottomup" "topdown" "random" "Contextual" "Concrete")

declare -a all_methods=("topdown")

declare -a all_seeds=(1 2 3 4 5) 


# declare -a all_data=("cifar100c")

# declare -a all_methods=("Contextual")

# declare -a all_seeds=(3) 


for data in "${all_data[@]}"
do

	for method in "${all_methods[@]}"
	do


		for seed in "${all_seeds[@]}"
		do
			echo $data
			echo $method 
			echo $seed
			sbatch Test_CIFAR_ResNet_corruption.sh $data $method $seed
		done
	done
done





###### test for transfer learning CIFAR10 and CIAFR100


# # ###GFN 
# declare -a all_data=("cifar10" "cifar100")


# declare -a all_methods=("bottomup" "topdown" "random" "Contextual" "Concrete")

# declare -a all_seeds=(1) 


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do


# 		for seed in "${all_seeds[@]}"
# 		do
# 			echo $data
# 			echo $method 
# 			echo $seed
# 			./Test_CIFAR_ResNet.sh $data $method $seed
# 		done
	
# 	done
# done