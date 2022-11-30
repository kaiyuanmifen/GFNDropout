#!/bin/bash
#SBATCH --job-name=VQA_GFN
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-10:50            # time (DD-HH:MM)


###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets



python3 ../vqa/run.py \
		--RUN='train' \
		--VERSION='GFN_bottomup' \
		--SPLIT='train' \
		--DP_TYPE=1 \
		--CONCRETE=0 \
		--LEARNPRIOR=1 \
		--DP_K=0.01 \
		--DP_ETA=-294 \
		--ARM=1 \
		--GPU='0' \
		--CTYPE="Bernoulli" \
		--add_noise=0 \
		--noise_scalar=5.0 \
		--GFN="bottomup"					