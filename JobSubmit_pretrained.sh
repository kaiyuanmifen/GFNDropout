#!/bin/bash
#SBATCH --job-name=dropout_pretrained
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=55G               # memory (per node)
#SBATCH --time=0-3:50            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets



###pretraining
#python Pretrain_MNIST.py --name "MNIST_Suprise"

Dropout=$1
seed=$2


# echo "data ${data}"
# echo "method ${method}"
# echo "depth ${depth}"

#name="${method}"_"${depth}"_"${data}"

#echo "name ${name}"
python Pretrained_model/train_cifar_pretrained.py \
--Dropout ${Dropout} \
--seed ${seed}
