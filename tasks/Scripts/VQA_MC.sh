#!/bin/bash
#SBATCH --job-name=VQA_MC
#SBATCH --gres=gpu:rtx8000:1               # Number of GPUs (per node)
#SBATCH --cpus-per-task=6
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-23:50            # time (DD-HH:MM)


###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets


SEED=$1


python3 ../vqa/run.py \
		--RUN='train' \
		--VERSION='mc_bernoulli2' \
		--SPLIT='train' \
		--DP_TYPE=0 \
		--CONCRETE=0 \
		--LEARNPRIOR=0 \
		--DP_K=0.01 \
		--DP_ETA=-294 \
		--ARM=0 \
		--GPU='0' \
		--CTYPE="Bernoulli" \
		--add_noise=0 \
		--noise_scalar=5.0 \
		--SEED=${SEED}						

