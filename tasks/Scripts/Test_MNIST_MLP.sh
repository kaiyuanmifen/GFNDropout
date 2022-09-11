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
										--load_file="../../checkpoints/ARMMLP_MNIST_ARMMLP_Contextual_20220910200949/best.model" \
										


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
										--load_file="../../checkpoints/ARMMLP_MNIST_ARMMLP_Contextual_20220910200949/best.model" \
										


#####concrete baseline



# python ../image_classification/main.py test \
# 										--model=ARMWideResNet \
# 										--GFN_dropout False \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--optimizer=momentum \
# 										--lr=0.1 \
# 										--schedule_milestone="[60, 120]" \
# 										--add_noisedata=False \
# 										--concretedp True \
# 										--dptype False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--augment_test=False \
# 										--model_name "_CIFAR_ARMWideResNet_Concrete" \
# 										--load_file="../../checkpoints/ARMWideResNet_CIFAR_ARMWideResNet_Concrete_20220905045434/best.model" \
										



# python ../image_classification/main.py test \
# 										--model=ARMWideResNet \
# 										--GFN_dropout False \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--optimizer=momentum \
# 										--lr=0.1 \
# 										--schedule_milestone="[60, 120]" \
# 										--add_noisedata=False \
# 										--concretedp True \
# 										--dptype False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--augment_test=True \
# 										--model_name "_CIFAR_ARMWideResNet_Concrete" \
# 										--load_file="../../checkpoints/ARMWideResNet_CIFAR_ARMWideResNet_Concrete_20220905045434/best.model" \
										





#####naive baseline



# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "both" \
# 										--lastlayer "NN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_both_NN_20220905220420/best.model" \
										




# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "both" \
# 										--lastlayer "NN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_both_NN_20220905220420/best.model" \
							

# ####MC  + NN baseline



python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dropout_rate 0.2 \
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
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_random_False_20220910203139/best.model" \
										




python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dropout_rate 0.2 \
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
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_random_False_20220910203139/best.model" \
										



# #####BNN baseline

# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "both" \
# 										--lastlayer "BNN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_both_BNN_20220905221322/best.model" \
										




# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "both" \
# 										--lastlayer "BNN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_both_BNN_20220905221322/best.model" \
										



# #####MC  + BNN baseline

# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "random" \
# 										--lastlayer "BNN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_random_BNN_20220906024339/best.model" \
										




# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "random" \
# 										--lastlayer "BNN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_random_BNN_20220906024339/best.model" \
										






# ####z mask and NN

# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "mu_mask" \
# 										--lastlayer "NN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_mu_mask_NN_20220906020137/best.model" \
										




# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "mu_mask" \
# 										--lastlayer "NN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_mu_mask_NN_20220906020137/best.model" \
				



# #####z mask only BNN

# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "mu_mask" \
# 										--lastlayer "BNN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_mu_mask_BNN_20220905221621/best.model" \
										




# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "mu_mask" \
# 										--lastlayer "BNN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_mu_mask_BNN_20220905221621/best.model" \
										



						




####topdown mask and NN

python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dropout_rate 0.2 \
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
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_topdown_False_20220910100521/best.model" \
										




python -u ../image_classification/main.py test \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dropout_rate 0.2 \
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
										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_topdown_False_20220910100521/best.model" \
										




# ####mu mask and BNN

# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "z_mask" \
# 										--lastlayer "BNN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_z_mask_BNN_20220905221330/best.model" \
										




# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "z_mask" \
# 										--lastlayer "BNN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_z_mask_BNN_20220905221330/best.model" \
										



# ######both mask and NN

# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "none" \
# 										--lastlayer "NN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_mu_mask_BNN_20220904220734/best.model" \
										




# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "none" \
# 										--lastlayer "NN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_none_BNN_20220904232954/best.model" \
										

# ######both mask and BNN

# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "none" \
# 										--lastlayer "BNN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_mu_mask_BNN_20220904220734/best.model" \
										




# python -u ../image_classification/main.py test \
# 										--model=MLP_GFN \
# 										--GFN_dropout True \
# 										--dropout_rate 0.2 \
# 										--dataset=mnist \
# 										--lambas='[.0,.0,.0,.0]' \
# 										--optimizer=adam \
# 										--lr=0.001 \
# 										--add_noisedata=False \
# 										--dptype False \
# 										--concretedp False \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--mask_off "none" \
# 										--lastlayer "BNN" \
# 										--model_name "_MNIST_MLP_GFN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/MLP_GFN_MNIST_MLP_GFN_none_BNN_20220904232954/best.model" \
										
