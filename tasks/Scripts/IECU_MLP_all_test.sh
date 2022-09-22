#!/bin/bash
#SBATCH --job-name=IECU_MLP
#SBATCH --gres=gpu:48gb:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=65G
#SBATCH --time=168:00:00         
#SBATCH --partition=long
#SBATCH --error=/home/mila/b/bonaventure.dossou/GFNDropout/slurmerror.txt
#SBATCH --output=/home/mila/b/bonaventure.dossou/GFNDropout/slurmoutput.txt

###########cluster information above this line
cd ../../../
module load python/3.6 cuda/10.1/cudnn/7.6 && source env/bin/activate
cd GFNDropout/tasks/Scripts/

python -u ../image_classification/main.py test \
										--model=ARMMLP \
										--GFN_dropout False \
										--dataset=iecu \
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
										--model_name "_IECU_ARMMLP_Concrete" \
										--load_file="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/ARMMLP_IECU_ARMMLP_Concrete_20220922170206/best.model" \


python -u ../image_classification/main.py test \
										--model=ARMMLP \
										--GFN_dropout False \
										--dataset=iecu \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--is_iecu True \
										--add_noisedata=False \
										--dptype True \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_IECU_ARMMLP_Contextual" \
										--load_file="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/ARMMLP_IECU_ARMMLP_Contextual_20220922170447/best.model" \

python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
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
										--model_name "_IECU_MLP_GFN" \
										--beta 0.001 \
										--load_file="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/MLP_GFN_IECU_MLP_GFN_random_False_20220922170850/best.model" \
										

python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
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
										--mask "none" \
										--BNN False \
										--model_name "_IECU_MLP_GFN" \
										--beta 0.001 \
										--max_epoch 100 \
										--load_file="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/MLP_GFN_IECU_MLP_GFN_none_False_20220922171133/best.model" \


python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
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
										--mask "topdown" \
										--BNN False \
										--model_name "_IECU_MLP_GFN" \
										--beta 0.001 \
										--load_file="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/MLP_GFN_IECU_MLP_GFN_topdown_False_20220922171358/best.model" \

python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--is_iecu True \
										--dropout_distribution 'bernoulli' \
										--mask "bottomup" \
										--BNN False \
										--model_name "_IECU_MLP_GFN" \
										--beta 0.001 \
										--load_file="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/MLP_GFN_IECU_MLP_GFN_bottomup_False_20220922171746/best.model" \

python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=iecu \
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
										--model_name "_IECU_MLP_GFN" \
										--beta 0.001 \
										--load_file="/home/mila/b/bonaventure.dossou/GFNDropout/checkpoints/MLP_GFN_IECU_MLP_GFN_upNdown_False_20220922172329/best.model" \