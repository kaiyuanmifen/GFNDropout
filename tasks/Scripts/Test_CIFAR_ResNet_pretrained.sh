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


#ciafar10


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
										--model_name "none_probing" \
										--mask "none" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_none_probing.model" 



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
										--model_name "none_probing" \
										--mask "none" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_none_probing.model" 


####random



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
										--model_name "random_probing" \
										--mask "random" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_random_probing.model" 



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
										--model_name "random_probing" \
										--mask "random" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_random_probing.model" 







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
										--model_name "random_tune" \
										--mask "random" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_random_tune.model" 



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
										--model_name "random_tune" \
										--mask "random" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_random_tune.model" 


 ####upNdown



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
										--model_name "upNdown_probing" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_upNdown_probing.model" 



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
										--model_name "upNdown_probing" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_upNdown_probing.model" 







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
										--model_name "upNdown_tune" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_upNdown_tune.model" 



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
										--model_name "upNdown_tune" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_upNdown_tune.model" 


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
										--model_name "topdown_probing" \
										--mask "topdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_topdown_probing.model" 



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
										--model_name "topdown_probing" \
										--mask "topdown" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_topdown_probing.model" 







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
										--model_name "topdown_tune" \
										--mask "topdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_topdown_tune.model" 



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
										--model_name "topdown_tune" \
										--mask "topdown" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_topdown_tune.model" 


 ##bottom up
 

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
										--model_name "bottomup_probing" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_bottomup_probing.model" 



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
										--model_name "bottomup_probing" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_bottomup_probing.model" 







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
										--model_name "bottomup_tune" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_bottomup_tune.model" 



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
										--model_name "bottomup_tune" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar10/CIFAR10_bottomup_tune.model" 







#####cifar 100

# #no mask tuning , probing


python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "none_probing" \
										--mask "none" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_none_probing.model" 



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "none_probing" \
										--mask "none" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_none_probing.model" 


####random



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "random_probing" \
										--mask "random" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_random_probing.model" 



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "random_probing" \
										--mask "random" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_random_probing.model" 







python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "random_tune" \
										--mask "random" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_random_tune.model" 



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "random_tune" \
										--mask "random" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_random_tune.model" 


 ####upNdown



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "upNdown_probing" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_upNdown_probing.model" 



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "upNdown_probing" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_upNdown_probing.model" 







python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "upNdown_tune" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_upNdown_tune.model" 



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "upNdown_tune" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_upNdown_tune.model" 


#topdown 



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "topdown_probing" \
										--mask "topdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_topdown_probing.model" 



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "topdown_probing" \
										--mask "topdown" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_topdown_probing.model" 







python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "topdown_tune" \
										--mask "topdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_topdown_tune.model" 



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "topdown_tune" \
										--mask "topdown" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_topdown_tune.model" 


 ##bottom up
 

python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "bottomup_probing" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_bottomup_probing.model" 



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "bottomup_probing" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_bottomup_probing.model" 







python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "bottomup_tune" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_bottomup_tune.model" 



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar100 \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "bottomup_tune" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/pretrained_cifar100/CIFAR100_bottomup_tune.model" 