#!/bin/bash
#SBATCH --job-name=VQA_contextual
#SBATCH --gres=gpu:rtx8000:1               # Number of GPUs (per node)
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-1:50            # time (DD-HH:MM)


###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets

SEED=$1
#setting inheritated from original authors, the naming is a bit confusing
"OOD version"
python3 ../vqa/run.py \
		--RUN='val' \
		--CKPT_V='bernoulli_contextual2' \
		--CKPT_E=13 \
		--DP_TYPE=1 \
		--CONCRETE=0 \
		--LEARNPRIOR=1 \
		--ARM=1 \
		--CTYPE="Bernoulli" \
		--DP_K=0.01 \
		--DP_ETA=-219 \
		--BS=128 \
		--UNCERTAINTY_SAMPLE=5 \
		--add_noise=1 \
		--noise_scalar=5.0 \
		--SEED=${SEED}												