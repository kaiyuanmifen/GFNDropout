# GFNDropout


############The paralell (faster) version of GFN dropout on MLP is call MLP_GFFN (F stands for faster)

To run it , you do , for example

python Run_training.py --Data "CIFAR10" --Method "MLP_GFFN" --Hidden_dim 1024 --p 0.5 --seed 42 --DataRatio 1.0 --Epochs 50 --RewardType 2

Hidden_dim=1024 (3 layers) is the standard setting from Hinton paper

In the code, "MNIST", "CIFAR10" and "SVHN" dataset are available 

RewardType 0: GFN reward usign training loss
RewardType 0: GFN reward using validation data 
RewardType 0: GFN reward using loss on augmented validaito data but mask generated using UNAUGMENTED validation data to force the GFN to pick invariant features.






###MasterFile.sh is the file to run all the codes and submit to Mila cluster 

just do ./MasterFile.sh

3 datasets have been implemented MINIST, CIFAR10 and SVHN 

two versions of GFN has been implemented detailed balanced (DB) or flow matching (FM). DB is much faster

When OODReward is set to 1, a combination of training loss and valiation loss using augmented data are used as reward for GFN 

Methods include 
1) no droput 
and the following working on only all NN layer
3) standard dropout 
4) Standout (Jimmy Ba method
5) Variational dropout (SVD) 
6) GFNDB
7) GFNFM
8)Resnet_FNDB

and the following working on all layers 

9) MLP_GFFN
10) "MLP_dropoutAll" 
11) "MLP_StandoutAll"
12) "MLP_SVDAll"
13)ConcreteDropout ( in progress)
