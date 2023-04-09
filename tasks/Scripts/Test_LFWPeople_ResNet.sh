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
										--dataset=LFWPeople \
										--add_noisedata=False \
										--concretedp True \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ARMWideResNet_Concrete" \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_Concrete.model" 


python ../image_classification/main.py test \
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=LFWPeople \
										--add_noisedata=False \
										--concretedp True \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ARMWideResNet_Concrete" \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_Concrete.model" 




python ../image_classification/main.py test \
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=LFWPeople \
										--add_noisedata=False \
										--concretedp True \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ARMWideResNet_Concrete" \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_Concrete.model" 


python ../image_classification/main.py test \
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=LFWPeople \
										--add_noisedata=False \
										--concretedp True \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ARMWideResNet_Concrete" \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_Concrete.model" 



# ###contextual baseline

python ../image_classification/main.py test \
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=LFWPeople \
										--lambas=.001 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--dptype True \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ARMWideResNet_Contextual" \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_Contextual.model" 



python ../image_classification/main.py test \
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=LFWPeople \
										--lambas=.001 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--dptype True \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ARMWideResNet_Contextual" \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_Contextual.model" 



python ../image_classification/main.py test \
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=LFWPeople \
										--lambas=.001 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--dptype True \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ARMWideResNet_Contextual" \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_Contextual.model" 



python ../image_classification/main.py test \
										--model=ResNet_Con \
										--GFN_dropout False \
										--dataset=LFWPeople \
										--lambas=.001 \
										--schedule_milestone="[60, 120]" \
										--add_noisedata=False \
										--dptype True \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ARMWideResNet_Contextual" \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_Contextual.model" 






# #no mask


python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "none" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_none.model" 





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "none" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_none.model" 



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "none" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_none.model" 





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "none" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_none.model" 

#random mask



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "random" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_random.model"  





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "random" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_random.model"  

python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "random" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_random.model"  





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "random" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_random.model"  


#upNdown



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_upNdown.model"  





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "none" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_upNdown.model"  


python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "upNdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_upNdown.model"  





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "none" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_upNdown.model"  



#topdown



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "topdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_topdown.model"  





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "topdown" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_topdown.model"  




python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "topdown" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_topdown.model"  





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "topdown" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_topdown.model"  


#bottomup


python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_bottomup.model"  





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_bottomup.model"   



python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=False \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_bottomup.model"  





python ../image_classification/main.py test \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=LFWPeople \
										--lambas=.001 \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_LFWPeople_ResNet_GFN" \
										--mask "bottomup" \
										--BNN False \
										--augment_test=True \
 										--load_file="../../../saved_checkpoint/LFWPeople/LFWPeople_bottomup.model"  