
import torch
import numpy as np

import torch.nn as nn

import torch.nn.functional as F

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

class DIY_Dropout(nn.Module):
    '''
    DIY version of Dropout, to make sure it is fair to compare with our method, torch offical implementation is computationally efficient but mathematically equivalent 
    '''
    def __init__(self, p=0.5):
        super(DIY_Dropout, self).__init__()
        self.p = p
        # multiplier is 1/(1-p). Set multiplier to 0 when p=1 to avoid error...
        if self.p < 1:
            self.multiplier_ = 1.0 / (1.0-p)
        else:
            self.multiplier_ = 0.0
    def forward(self, input):
        # if model.eval(), don't apply dropout
        if not self.training:
            return input
        
        # So that we have `input.shape` numbers of Bernoulli(1-p) samples
        selected_ = torch.Tensor(input.shape).uniform_(0,1)>self.p
        
        # To support both CPU and GPU.
        if input.is_cuda:
            selected_ = Variable(selected_.type(torch.cuda.FloatTensor), requires_grad=False)
        else:
            selected_ = Variable(selected_.type(torch.FloatTensor), requires_grad=False)
            
        # Multiply output by multiplier as described in the paper [1]
        return torch.mul(selected_,input) * self.multiplier_




class Mask_Dropout(nn.Module):
    '''
    Mask_Dropout, that takes in a mask to decide which unit to drop
    '''
    def __init__(self):
        super(Mask_Dropout, self).__init__()
   
    def forward(self, input,mask):
        # if model.eval(), don't apply dropout
        ###this dropout is always on, even in evaluation, there need to aggregate samples from the inference

        # if not self.training:
        #     print("Still using dropout in evaluation")
            

 
        p=1-1.0*mask.sum()/(mask.shape[0]*mask.shape[1])#% of unit dropped

        selected_ = mask #mask is a binary vector with the same shape as input, where 1 indicate not dropping and 0 indicates dropping
        

        # multiplier is 1/(1-p). Set multiplier to 0 when p=1 to avoid error...
        if p < 1:
            self.multiplier_ = 1.0 / (1.0-p)
        else:
            self.multiplier_ = 0.0

        
        
        # To support both CPU and GPU.
        if input.is_cuda:
            selected_ = Variable(selected_.type(torch.cuda.FloatTensor), requires_grad=False)
        else:
            selected_ = Variable(selected_.type(torch.FloatTensor), requires_grad=False)
            
        # Multiply output by multiplier as described in the paper [1]
    
        return torch.mul(selected_,input) * self.multiplier_


if __name__ == "__main__":
    
    bz=12
    dim=9
    Input=torch.randn((bz,dim))

    ####original dropout

    dropout=DIY_Dropout(0.3)

    output=dropout(Input)

    # print("original output")
    # print(output)

    #####dropout with mask provided 
    dropout=Mask_Dropout()
    mask=torch.zeros((bz,dim)).uniform_(0,1)>0.9
    
    output=dropout(Input,mask)

    print("masked output")
    print(output)