#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-2:50            # time (DD-HH:MM)


###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
#conda activate GNN
conda activate GFlownets



#########MLP

# #####comtextual baseline

python -u ../image_classification/main.py test \
										--model=ARMMLP \
										--GFFN_dropout False \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype True \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--augment_test=False \
										--model_name "_MNIST_ARMMLP_Contextual" \
										--load_file="../../checkpoints/ARMMLP_MNIST_ARMMLP_Contextual_20220916142356/190.model" \
										


python -u ../image_classification/main.py test \
										--model=ARMMLP \
										--GFFN_dropout False \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype True \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--augment_test=True \
										--model_name "_MNIST_ARMMLP_Contextual" \
										--load_file="../../checkpoints/ARMMLP_MNIST_ARMMLP_Contextual_20220916142356/190.model" \
										


#####concrete baseline



python ../image_classification/main.py test \
										--model=ARMWideResNet \
										--GFN_dropout False \
										--dataset=cifar10 \
										--lambas=.001 \
										--optimizer=momentum \
										--lr=0.1 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--concretedp True \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--augment_test=False \
										--model_name "_CIFAR_ARMWideResNet_Concrete" \
										--load_file="../../checkpoints/ARMMLP_MNIST_ARMMLP_Concrete_20220916142248/190.model" \
										



python ../image_classification/main.py test \
										--model=ARMWideResNet \
										--GFN_dropout False \
										--dataset=cifar10 \
										--lambas=.001 \
										--optimizer=momentum \
										--lr=0.1 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--concretedp True \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--augment_test=True \
										--model_name "_CIFAR_ARMWideResNet_Concrete" \
										--load_file="../../checkpoints/ARMMLP_MNIST_ARMMLP_Concrete_20220916142248/190.model" \
										








####no mask  + NN baseline



python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "none" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--augment_test=False \
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_none_False_20220916142854/190.model" \
										




python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "none" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--augment_test=True \
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_none_False_20220916142854/190.model" \
										


####MC  + NN baseline



python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "random" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--augment_test=False \
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_random_False_20220916142541/190.model" \
										




python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "random" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--augment_test=True \
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_random_False_20220916142541/190.model" \
										







####topdown mask and NN

python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "topdown" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--augment_test=False \
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_topdown_False_20220916142541/190.model" \
										




python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "topdown" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--augment_test=True \
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_topdown_False_20220916142541/190.model" \
										



####bottom up and NN

python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "bottomup" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--augment_test=False \
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_bottomup_False_20220916142250/190.model" \
										




python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "bottomup" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--augment_test=True \
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_bottomup_False_20220916142250/190.model" \
										


####upNdown mask and NN

python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "upNdown" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--augment_test=False \
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_upNdown_False_20220916142350/190.model" \
										




python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dataset=mnist \
										--lambas='[.0,.0,.0,.0]' \
										--optimizer=adam \
										--lr=0.001 \
										--add_noisedata=False \
										--dptype False \
										--concretedp False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--mask "upNdown" \
										--BNN False \
										--model_name "_MNIST_MLP_GFN" \
										--augment_test=True \
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_upNdown_False_20220916142350/190.model" \
										



