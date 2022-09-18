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


####concrete baseline

python ../image_classification/main.py test \
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=cifar10 \
										--add_noisedata=False \
										--concretedp True \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ARMWideResNet_Concrete" \
										--augment_test=False \
 										--load_file="../../checkpoints/ResNet_Con_CIFAR_ARMWideResNet_Concrete_20220917130756/150.model" 


python ../image_classification/main.py test \
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=cifar10 \
										--add_noisedata=False \
										--concretedp True \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ARMWideResNet_Concrete" \
										--augment_test=True \
 										--load_file="../../checkpoints/ResNet_Con_CIFAR_ARMWideResNet_Concrete_20220917130756/150.model" 



###contextual baseline

python ../image_classification/main.py test \
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=cifar10 \
										--lambas=.001 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--dptype True \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ARMWideResNet_Contextual" \
										--augment_test=False \
 										--load_file="../../checkpoints/ResNet_Con_CIFAR_ARMWideResNet_Contextual_20220917130754/150.model" 



python ../image_classification/main.py test \
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=cifar10 \
										--lambas=.001 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--dptype True \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ARMWideResNet_Contextual" \
										--augment_test=True \
 										--load_file="../../checkpoints/ResNet_Con_CIFAR_ARMWideResNet_Contextual_20220917130754/150.model" 





#no mask


python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "none" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../checkpoints/ResNet_GFN_CIFAR_ResNet_GFN_none_False_20220917130457/150.model" 





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "none" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../checkpoints/ResNet_GFN_CIFAR_ResNet_GFN_none_False_20220917130457/150.model" 



#random mask



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "random" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../checkpoints/ResNet_GFN_CIFAR_ResNet_GFN_random_False_20220917023958/150.model" 





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "random" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../checkpoints/ResNet_GFN_CIFAR_ResNet_GFN_random_False_20220917023958/150.model" 



#upNdown



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../checkpoints/ResNet_GFN_CIFAR_ResNet_GFN_upNdown_False_20220917023659/150.model" 





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "none" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../checkpoints/ResNet_GFN_CIFAR_ResNet_GFN_upNdown_False_20220917023659/150.model" 



#topdown



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "topdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../checkpoints/ResNet_GFN_CIFAR_ResNet_GFN_topdown_False_20220917023655/150.model" 





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "topdown" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../checkpoints/ResNet_GFN_CIFAR_ResNet_GFN_topdown_False_20220917023655/150.model" 




#bottomup


python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../checkpoints/ResNet_GFN_CIFAR_ResNet_GFN_bottomup_False_20220917023655/150.model" 





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../checkpoints/ResNet_GFN_CIFAR_ResNet_GFN_bottomup_False_20220917023655/150.model" 

