
import torch
import numpy as np
import torch.optim as optim 
from GFN_SampleMask import GFN_SamplingMask

from GFNFunctions import *
from Dropout_DIY import *
from TaskModels import *


import matplotlib.pyplot as plt
import matplotlib

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import random

import numpy as np
import pandas as pd
import time
import h5py
from scipy.ndimage.interpolation import rotate

import argparse

matplotlib.use('AGG')





#####zero eptoch
Data=torch.load("Results/pretrained_True_LinearProbing_GFNZeroEpoch_1testmask.pt",map_location=torch.device('cpu'))
print(Data.shape)

print(Data[1,10,:])

print((Data.sum(2)/Data.shape[2])[0,:])

MeanPercentage=(Data.sum(2)/Data.shape[2]).mean().item()

VecPlot=(Data.sum(2)/Data.shape[2])[2,:].tolist()


fig=plt.figure(figsize=(8, 7))

# ax = fig.add_axes([0,0,1,1])
# ax.bar(range(1, len(VecPlot)+1), VecPlot,0.1)

# ax.set_ylabel('% dropout')

# ax.set_xlabel('unit in the model')
# ax.set_title('Sampled dropout from model')

plt.plot(range(1, len(VecPlot)+1), VecPlot, '.', alpha=0.6);
plt.ylim([-0.1, 1.2]);
#plt.legend(loc=1);
plt.xlabel('unit');
plt.ylabel('% dropout over 5 repeats');
plt.title("at 0 epochs on average "+str(100*MeanPercentage)+'% of dropout of \n a single data point over 5 samples of masks')
plt.savefig('Images/MasksSamples_0_epoch.png')
plt.clf()



####across different data points

Y=[]
X=[]


for i in range(100):
	Y=Y+(Data.sum(2)/Data.shape[2])[i,:].tolist()
	X=X+list(range(Data.shape[1]))


plt.plot(X, Y, '.', alpha=0.6);
plt.ylim([-0.1, 1.2]);
#plt.legend(loc=1);
plt.xlabel('unit');
plt.ylabel('% dropout over 5 repeats');
plt.title("All data points, After 0 epochs on average "+str(100*MeanPercentage)+'% of dropout of \n a single data point over 5 samples of masks')
plt.savefig('Images/AllDatapoints_MasksSamples_0_epoch.png')
plt.clf()


#####2 eptoch
Data=torch.load("Results/pretrained_True_LinearProbing_GFNintermediate_1testmask.pt",map_location=torch.device('cpu'))
print(Data.shape)

print(Data[1,10,:])

print((Data.sum(2)/Data.shape[2])[0,:])

MeanPercentage=(Data.sum(2)/Data.shape[2]).mean().item()

VecPlot=(Data.sum(2)/Data.shape[2])[2,:].tolist()


fig=plt.figure(figsize=(8, 7))

# ax = fig.add_axes([0,0,1,1])
# ax.bar(range(1, len(VecPlot)+1), VecPlot,0.1)

# ax.set_ylabel('% dropout')

# ax.set_xlabel('unit in the model')
# ax.set_title('Sampled dropout from model')

plt.plot(range(1, len(VecPlot)+1), VecPlot, '.', alpha=0.6);
plt.ylim([-0.1, 1.2]);
#plt.legend(loc=1);
plt.xlabel('unit');
plt.ylabel('% dropout over 5 repeats');
plt.title("at 2 epochs on average "+str(100*MeanPercentage)+'% of dropout of \n a single data point over 5 samples of masks')
plt.savefig('Images/MasksSamples_2_epoch.png')
plt.clf()



####across different data points

Y=[]
X=[]


for i in range(100):
	Y=Y+(Data.sum(2)/Data.shape[2])[i,:].tolist()
	X=X+list(range(Data.shape[1]))


plt.plot(X, Y, '.', alpha=0.6);
plt.ylim([-0.1, 1.2]);
#plt.legend(loc=1);
plt.xlabel('unit');
plt.ylabel('% dropout over 5 repeats');
plt.title("All data points, After 2 epochs on average "+str(100*MeanPercentage)+'% of dropout of \n a single data point over 5 samples of masks')
plt.savefig('Images/AllDatapoints_MasksSamples_2_epoch.png')
plt.clf()

#####15 eptoch
Data=torch.load("Results/pretrained_True_LinearProbing_GFN15epoch_1testmask.pt",map_location=torch.device('cpu'))
print(Data.shape)

print(Data[1,10,:])

print((Data.sum(2)/Data.shape[2])[0,:])

MeanPercentage=(Data.sum(2)/Data.shape[2]).mean().item()

VecPlot=(Data.sum(2)/Data.shape[2])[2,:].tolist()


fig=plt.figure(figsize=(8, 7))

# ax = fig.add_axes([0,0,1,1])
# ax.bar(range(1, len(VecPlot)+1), VecPlot,0.1)

# ax.set_ylabel('% dropout')

# ax.set_xlabel('unit in the model')
# ax.set_title('Sampled dropout from model')

plt.plot(range(1, len(VecPlot)+1), VecPlot, '.', alpha=0.6);
plt.ylim([-0.1, 1.2]);
#plt.legend(loc=1);
plt.xlabel('unit');
plt.ylabel('% dropout over 5 repeats');
plt.title("at 15 epochs on average "+str(100*MeanPercentage)+'% of dropout of \n a single data point over 5 samples of masks')
plt.savefig('Images/MasksSamples_15_epoch.png')
plt.clf()



####across different data points

Y=[]
X=[]


for i in range(100):
	Y=Y+(Data.sum(2)/Data.shape[2])[i,:].tolist()
	X=X+list(range(Data.shape[1]))


plt.plot(X, Y, '.', alpha=0.6);
plt.ylim([-0.1, 1.2]);
#plt.legend(loc=1);
plt.xlabel('unit');
plt.ylabel('% dropout over 5 repeats');
plt.title("All data points, After 15 epochs on average "+str(100*MeanPercentage)+'% of dropout of \n a single data point over 5 samples of masks')
plt.savefig('Images/AllDatapoints_MasksSamples_15_epoch.png')
plt.clf()


#####after train 200 epoch

Data=torch.load("Results/pretrained_True_LinearProbing_GFNDebugging_1testmask_aftertrain.pt",map_location=torch.device('cpu'))
print(Data.shape)

print(Data[1,10,:])

print((Data.sum(2)/Data.shape[2])[0,:])

MeanPercentage=(Data.sum(2)/Data.shape[2]).mean().item()

VecPlot=(Data.sum(2)/Data.shape[2])[100,:].tolist()


fig=plt.figure(figsize=(8, 7))

# ax = fig.add_axes([0,0,1,1])
# ax.bar(range(1, len(VecPlot)+1), VecPlot,0.1)

# ax.set_ylabel('% dropout')

# ax.set_xlabel('unit in the model')
# ax.set_title('Sampled dropout from model')

plt.plot(range(1, len(VecPlot)+1), VecPlot, '.', alpha=0.6);
plt.ylim([-0.1, 1.2]);
#plt.legend(loc=1);
plt.xlabel('unit');
plt.ylabel('% dropout over 5 repeats');
plt.title("After 200 epochs on average "+str(100*MeanPercentage)+'% of dropout of \n a single data point over 5 samples of masks')
plt.savefig('Images/MasksSamples_200_epoch.png')
plt.clf()

####across different data points

Y=[]
X=[]


for i in range(100):
	Y=Y+(Data.sum(2)/Data.shape[2])[i,:].tolist()
	X=X+list(range(Data.shape[1]))


plt.plot(X, Y, '.', alpha=0.6);
plt.ylim([-0.1, 1.2]);
#plt.legend(loc=1);
plt.xlabel('unit');
plt.ylabel('% dropout over 5 repeats');
plt.title("All data points, After 200 epochs on average "+str(100*MeanPercentage)+'% of dropout of \n a single data point over 5 samples of masks')
plt.savefig('Images/AllDatapoints_MasksSamples_200_epoch.png')
plt.clf()