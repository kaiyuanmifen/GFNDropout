# GFNDropout


###MasterFile.sh is the file to run all the codes 

just do ./MasterFile.sh

3 datasets have been implemented MINIST, CIFAR10 and SVHN 

two versions of GFN has been implemented detailed balanced (DB) or flow matching (FM). DB is much faster

When OODReward is set to 1, a combination of training loss and valiation loss using augmented data are used as reward for GFN 

Methods include 
1) no droput 
2) standard dropout 
3) Standout (Jimmy Ba method
4) Variational dropout (SVD) 
5) GFNDB
6) GFNFM
