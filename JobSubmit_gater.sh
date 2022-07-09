#!/bin/bash
#SBATCH --job-name=dropout_gater
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=85G               # memory (per node)
#SBATCH --time=0-14:50            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets



###pretraining
#python Pretrain_MNIST.py --name "MNIST_Suprise"

data=$1

method=$2

depth=$3

echo "data ${data}"
echo "method ${method}"
echo "depth ${depth}"

name="${method}"_"${depth}"_"${data}"

echo "name ${name}"
python gaternet/main_train.py \
--backbone-depth $depth \
--data-dir ../../../data/ \
--method $method \
--name $name