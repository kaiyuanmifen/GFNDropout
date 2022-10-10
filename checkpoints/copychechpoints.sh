#!/bin/bash
#odinary version

# for filename in ./* ; do
#     echo "$filename"		
#     echo ${filename##*_}
#     seed=${filename##*_}
#     ###decide folder
#     if [[ $filename == *"cifar10_"* ]];then
#     	folder="cifar10"
#     	echo $folder
#     fi

#     if [[ $filename == *"cifar100_"* ]];then
#     	folder="cifar100"
#     	echo $folder
#     fi
 
#     if [[ $filename == *"mnist_"* ]];then
#     	folder="MNIST"
#     	echo $folder
#     fi

#     ##decide name
#     if [[ $filename == *"random_"* ]];then
# 		method="random"	
#     fi

#     if [[ $filename == *"none_"* ]];then
# 		method="none"
#     fi
 
#     if [[ $filename == *"topdown_"* ]];then
# 		method="topdown"
#     fi		


#     if [[ $filename == *"bottomup_"* ]];then
# 		method="bottomup"
#     fi	

#     if [[ $filename == *"Contextual_"* ]];then
# 		method="Contextual"
#     fi	

#     if [[ $filename == *"Concrete_"* ]];then
# 		method="Concrete"
#     fi
#  	echo $method

#  	###copy file 

#  	cp "${filename}/best.model" "/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/saved_checkpoint/${folder}/${folder}_${method}_${seed}.model"
# done


#pretrained version

# for filename in ./* ; do
#     echo "$filename"        
#     echo ${filename##*_}
#     seed=${filename##*_}
#     ###decide folder
#     if [[ $filename == *"cifar10_"* ]];then
#         folder="pretrained_cifar10"
#         echo $folder
#     fi

#     if [[ $filename == *"cifar100_"* ]];then
#         folder="pretrained_cifar100"
#         echo $folder
#     fi
 
#     if [[ $filename == *"mnist_"* ]];then
#         folder="pretrained_MNIST"
#         echo $folder
#     fi

#     ##decide name
#     if [[ $filename == *"random_"* ]];then
#         method="random" 
#     fi

#     if [[ $filename == *"none_"* ]];then
#         method="none"
#     fi
 
#     if [[ $filename == *"topdown_"* ]];then
#         method="topdown"
#     fi      


#     if [[ $filename == *"bottomup_"* ]];then
#         method="bottomup"
#     fi  

#     if [[ $filename == *"Contextual_"* ]];then
#         method="Contextual"
#     fi  

#     if [[ $filename == *"Concrete_"* ]];then
#         method="Concrete"
#     fi
#     echo $method

#     ###copy file 

#     cp "${filename}/best.model" "/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/saved_checkpoint/${folder}/${folder}_${method}_${seed}.model"
# done



####transfer


for filename in ./* ; do
    echo "$filename"        
    echo ${filename##*_}
    seed=${filename##*_}
    subsetsize=${filename##*_}
    ###decide folder
    if [[ $filename == *"cifar10_"* ]];then
        folder="transfer_cifar10"
        echo $folder
    fi

    if [[ $filename == *"cifar100_"* ]];then
        folder="transfer_cifar100"
        echo $folder
    fi
 
    if [[ $filename == *"mnist_"* ]];then
        folder="pretrained_MNIST"
        echo $folder
    fi

    ##decide name
    if [[ $filename == *"random_"* ]];then
        method="random" 
    fi

    if [[ $filename == *"none_"* ]];then
        method="none"
    fi
 
    if [[ $filename == *"topdown_"* ]];then
        method="topdown"
    fi      


    if [[ $filename == *"bottomup_"* ]];then
        method="bottomup"
    fi  

    if [[ $filename == *"Contextual_"* ]];then
        method="Contextual"
    fi  

    if [[ $filename == *"Concrete_"* ]];then
        method="Concrete"
    fi
    echo $method

    ###copy file 

    cp "${filename}/best.model" "/home/mila/d/dianbo.liu/scratch/GFlownets/Dropout/saved_checkpoint/${folder}/${folder}_${method}_${seed}.model"
done


