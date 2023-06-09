# GFNDropout

This is the code for GFN dropout project , currently it include codes for MLP and resnet for MNIST,CIFAR
dataset and transformer for VQA task

This code is partially adapted from :https://github.com/szhang42/Contextual_dropout_release

TO run the expeirment go to GFNDropout/tasks/Scripts/ and run corresponding experiment 

for example to run GFN dropout using MLP model on MNIST with "topdown" mask without BNN backbone 

the two important settings are "--mask" which types of mask to use and "--BNN" whether to use BNN as backbone

python -u ../image_classification/main.py train \
										--model=MLP_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
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
										--max_epoch 200 \
                    
run GFN dropout using Resnet18 model on cifar with "topdown" mask without BNN backbone 


 python ../image_classification/main.py train \
										--model=ResNet_GFN \
										--GFN_dropout True \
										--dropout_rate 0.5 \
										--dataset=cifar10 \
										--lambas=.001 \
										--optimizer=momentum \
										--lr=0.1 \
										--schedule_milestone="[25, 40]" \
										--add_noisedata=False \
										--concretedp False \
										--dptype False \
										--fixdistrdp False \
										--ctype "Bernoulli" \
										--dropout_distribution 'bernoulli' \
										--model_name "_CIFAR_ResNet_GFN" \
										--mask "topdown" \
										--BNN False \
										--max_epoch 200 \
                    
 To test performance on augmented data :
 

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
										--load_file="your model file location" \
										
