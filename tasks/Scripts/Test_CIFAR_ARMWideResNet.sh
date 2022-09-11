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







# #contextual baseline

# python ../image_classification/main.py test \
# 										--model=ARMWideResNet \
# 										--GFN_dropout False \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--optimizer=momentum \
# 										--lr=0.1 \
# 										--schedule_milestone="[60, 120]" \
# 										--add_noisedata=False \
# 										--dptype True \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--model_name "_CIFAR_ARMWideResNet_Contextual" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/ARMWideResNet_CIFAR_ARMWideResNet_Contextual_20220831141213/best.model" 




# python ../image_classification/main.py test \
# 										--model=ARMWideResNet \
# 										--GFN_dropout False \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--optimizer=momentum \
# 										--lr=0.1 \
# 										--schedule_milestone="[60, 120]" \
# 										--add_noisedata=False \
# 										--dptype True \
# 										--fixdistrdp False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--model_name "_CIFAR_ARMWideResNet_Contextual" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/ARMWideResNet_CIFAR_ARMWideResNet_Contextual_20220831141213/best.model" 


# ####Resnet Baseline 



#  python ../image_classification/main.py test \
# 										--model=ARMWideResNet_GFN \
# 										--GFN_dropout True \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--add_noisedata=False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--model_name "_CIFAR_ARMWideResNet_GFN" \
# 										--mask_off "both" \
# 										--lastlayer "NN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/ARMWideResNet_GFN_CIFAR_ARMWideResNet_GFN_both_NN_base/best.model" 


#  python ../image_classification/main.py test \
# 										--model=ARMWideResNet_GFN \
# 										--GFN_dropout True \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--add_noisedata=False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--model_name "_CIFAR_ARMWideResNet_GFN" \
# 										--mask_off "both" \
# 										--lastlayer "NN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/ARMWideResNet_GFN_CIFAR_ARMWideResNet_GFN_both_NN_base/best.model" 


# #####both mask on



#  python ../image_classification/main.py test \
# 										--model=ARMWideResNet_GFN \
# 										--GFN_dropout True \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--add_noisedata=False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--model_name "_CIFAR_ARMWideResNet_GFN" \
# 										--mask_off "none" \
# 										--lastlayer "NN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/ARMWideResNet_GFN_CIFAR_ARMWideResNet_GFN_none_NN_20220831224808/best.model" 


#  python ../image_classification/main.py test \
# 										--model=ARMWideResNet_GFN \
# 										--GFN_dropout True \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--add_noisedata=False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--model_name "_CIFAR_ARMWideResNet_GFN" \
# 										--mask_off "none" \
# 										--lastlayer "NN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/ARMWideResNet_GFN_CIFAR_ARMWideResNet_GFN_none_NN_20220831224808/best.model" 






# #####mu mask only



#  python ../image_classification/main.py test \
# 										--model=ARMWideResNet_GFN \
# 										--GFN_dropout True \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--add_noisedata=False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--model_name "_CIFAR_ARMWideResNet_GFN" \
# 										--mask_off "z_mask" \
# 										--lastlayer "NN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/ARMWideResNet_GFN_CIFAR_ARMWideResNet_GFN_z_mask_NN_20220831225408/best.model" 


#  python ../image_classification/main.py test \
# 										--model=ARMWideResNet_GFN \
# 										--GFN_dropout True \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--add_noisedata=False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--model_name "_CIFAR_ARMWideResNet_GFN" \
# 										--mask_off "z_mask" \
# 										--lastlayer "NN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/ARMWideResNet_GFN_CIFAR_ARMWideResNet_GFN_z_mask_NN_20220831225408/best.model" 




# #####z mask only



#  python ../image_classification/main.py test \
# 										--model=ARMWideResNet_GFN \
# 										--GFN_dropout True \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--add_noisedata=False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--model_name "_CIFAR_ARMWideResNet_GFN" \
# 										--mask_off "mu_mask" \
# 										--lastlayer "NN" \
# 										--augment_test=True \
# 										--load_file="../../checkpoints/ARMWideResNet_GFN_CIFAR_ARMWideResNet_GFN_mu_mask_NN_20220831230012/best.model" 


#  python ../image_classification/main.py test \
# 										--model=ARMWideResNet_GFN \
# 										--GFN_dropout True \
# 										--dataset=cifar10 \
# 										--lambas=.001 \
# 										--add_noisedata=False \
# 										--ctype "Bernoulli" \
# 										--dropout_distribution 'bernoulli' \
# 										--model_name "_CIFAR_ARMWideResNet_GFN" \
# 										--mask_off "mu_mask" \
# 										--lastlayer "NN" \
# 										--augment_test=False \
# 										--load_file="../../checkpoints/ARMWideResNet_GFN_CIFAR_ARMWideResNet_GFN_mu_mask_NN_20220831230012/best.model" 



