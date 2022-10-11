#!/bin/bash
#SBATCH --job-name=IECU_MLP_AL
#SBATCH --gres=gpu:48gb:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=65G
#SBATCH --time=168:00:00         
#SBATCH --partition=long
#SBATCH --error=/home/mila/b/bonaventure.dossou/GFNDropout/al_slurmerror.txt
#SBATCH --output=/home/mila/b/bonaventure.dossou/GFNDropout/al_slurmoutput.txt

###########cluster information above this line
cd ../../../
module load python/3.6 cuda/10.1/cudnn/7.6 && source env/bin/activate
cd GFNDropout/tasks/Scripts/

python -u ../image_classification/main.py active_learning \
										--model=ARMMLP \
										--GFN_dropout False \
										--dataset=iecu_al \
										--al_rounds=10\
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--is_iecu True \
										--concretedp True \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_AL_IECU_ARMMLP_Concrete" \
										--max_epoch 15 \
										# --start_from="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/ARMMLP_IECU_ARMMLP_Concrete_20220923221017/best.model" \

python -u ../image_classification/main.py active_learning \
										--model=ARMMLP \
										--GFN_dropout False \
										--dataset=iecu_al \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--is_iecu True \
										--add_noisedata=False \
										--dptype True \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--al_rounds=10\
										--dropout_distribution 'bernoulli' \
										--model_name "_AL_IECU_ARMMLP_Contextual" \
										--max_epoch 15 \
										# --start_from="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/ARMMLP_IECU_ARMMLP_Contextual_20220924001210/best.model" \

python -u ../image_classification/main.py active_learning \
										--model=MLP_GFN \
										--GFN_dropout True \
										--al_rounds=10\
										--dataset=iecu_al \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--is_iecu True \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "random" \
										--BNN False \
										--model_name "_AL_IECU_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 15 \
										# --start_from="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/MLP_GFN_IECU_MLP_GFN_random_False_20220924013941/best.model" \
										

python -u ../image_classification/main.py active_learning \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu_al \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--is_iecu True \
										--al_rounds=10\
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "none" \
										--BNN False \
										--model_name "_AL_IECU_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 15 \
										# --start_from="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/MLP_GFN_IECU_MLP_GFN_none_False_20220924025457/best.model" \

python -u ../image_classification/main.py active_learning \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu_al \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--al_rounds=10\
										--dptype False \
										--is_iecu True \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "topdown" \
										--BNN False \
										--model_name "_AL_IECU_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 15 \
										# --start_from="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/MLP_GFN_IECU_MLP_GFN_topdown_False_20220924092129/best.model" \


python -u ../image_classification/main.py active_learning \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu_al \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--al_rounds=10\
										--is_iecu True \
										--dropout_distribution 'bernoulli' \
										--mask "bottomup" \
										--BNN False \
										--model_name "_AL_IECU_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 15 \
										# --start_from="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/MLP_GFN_IECU_MLP_GFN_bottomup_False_20220924143304/best.model" \

python -u ../image_classification/main.py active_learning \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu_al \
										--al_rounds=10\
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--is_iecu True \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "upNdown" \
										--BNN False \
										--model_name "_AL_IECU_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 15 \
										# --start_from="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/MLP_GFN_IECU_MLP_GFN_upNdown_False_20220924164117/best.model" \