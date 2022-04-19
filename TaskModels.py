
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
from torch.nn import Parameter
from Dropout_DIY import *
from resnet import Block

class MLP(nn.Module):
    def __init__(self,Input_size, hidden_size=10, droprates=0):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(Input_size,4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)


        self.DIY_Dropout=DIY_Dropout(droprates)###make it part of the model so it gets the train/eval state

        self.droprates=droprates

    def forward(self, x):
      
        x = x.view(x.shape[0], -1)
        #x=DIY_Dropout(p=self.droprates[0])(x)
        x=F.relu(self.fc1(x))
        
        x=F.relu(self.fc2(x))
        x=self.DIY_Dropout(x)
        #x=DIY_Dropout(p=self.droprates[1])(x)
        x=self.fc3(x)
        return x


class MLP_MaskedDropout(nn.Module):
    def __init__(self,Input_size, hidden_size=10):
        super(MLP_MaskedDropout, self).__init__()

  
        self.fc1 = nn.Linear(Input_size,4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

        self.Mask_Dropout=Mask_Dropout()###make it part of the model so it gets the train/eval state


    def forward(self,x,mask):
      
        x = x.view(x.shape[0], -1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.Mask_Dropout(x,mask)
        #x=Mask_Dropout()(x,masks[1])
        x=self.fc3(x)
        return x

    def Get_condition(self,x):

        x = x.view(x.shape[0], -1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))

        return x



####sparse variational dropout
class LinearSVDO(nn.Module):
    def __init__(self, in_features, out_features, threshold, bias=True):
        super(LinearSVDO, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        self.log_sigma.data.fill_(-5)        
        
    def forward(self, x):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10) 
        
        if self.training:
            lrt_mean =  F.linear(x, self.W) + self.bias
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps
    
        return F.linear(x, self.W * (self.log_alpha < 3).float()) + self.bias
        
    def kl_reg(self):
        # Return KL here -- a scalar 
        k1, k2, k3 = torch.Tensor([0.63576]).cuda(), torch.Tensor([1.8732]).cuda(), torch.Tensor([1.48695]).cuda()
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        a = - torch.sum(kl)
        return a


class MLP_SVD(nn.Module):
    def __init__(self, Input_size,hidden_size=10,threshold=3):
        super(MLP_SVD, self).__init__()

        #self.fc1 = nn.Linear(28*28,hidden_size)
        self.fc1 = nn.Linear(Input_size,4*hidden_size)
        self.fc2 = LinearSVDO(4*hidden_size, hidden_size, threshold) 
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
      
        x = x.view(x.shape[0], -1)
        #x=DIY_Dropout(p=self.droprates[0])(x)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        #x=DIY_Dropout(p=self.droprates[1])(x)
        x=self.fc3(x)
        return x

####standout 
from torch.autograd import Variable
from torch import nn

class Standout(nn.Module):

    def __init__(self, last_layer, alpha, beta):
        print("<<<<<<<<< THIS IS DEFINETLY A STANDOUT TRAINING >>>>>>>>>>>>>>>")
        super(Standout, self).__init__()
        self.pi = last_layer.weight
        self.alpha = alpha
        self.beta = beta
        self.nonlinearity = nn.Sigmoid()


    def forward(self, previous, current, p=0.5, deterministic=False):
        # Function as in page 3 of paper: Variational Dropout
        self.p = self.nonlinearity(self.alpha * previous.matmul(self.pi.t()) + self.beta)
        self.mask = sample_mask(self.p)

        # Deterministic version as in the paper
        if(deterministic or torch.mean(self.p).data.cpu().numpy()==0):
            return self.p * current
        else:
            return self.mask * current

def sample_mask(p):
    """Given a matrix of probabilities, this will sample a mask in PyTorch."""

    if torch.cuda.is_available():
        uniform = Variable(torch.Tensor(p.size()).uniform_(0, 1).cuda())
    else:
        uniform = Variable(torch.Tensor(p.size()).uniform_(0, 1))
    mask = uniform < p

    if torch.cuda.is_available():
        mask = mask.type(torch.cuda.FloatTensor)
    else:
        mask = mask.type(torch.FloatTensor)

    return mask




class MLP_Standout(nn.Module):
    def __init__(self,Input_size, hidden_size=10,droprates=0.5):
        super(MLP_Standout, self).__init__()

        self.fc1 = nn.Linear(Input_size,4*hidden_size)
        
        
        self.fc2 = nn.Linear(4*hidden_size,hidden_size)
        
        self.fc2_drop = Standout(self.fc2, droprates, 1)


        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
      
        x = x.view(x.shape[0], -1)
        #x=DIY_Dropout(p=self.droprates[0])(x)
        
        x= F.relu(self.fc1(x))

        previous = x
        x_relu = F.relu(self.fc2(x))
        # Select between dropouts styles
        x = self.fc2_drop(previous, x_relu)
        #x=DIY_Dropout(p=self.droprates[1])(x)
        x=self.fc3(x)

        return x


##########CNN code 
class CNN(nn.Module):
    def __init__(self,image_size,hidden_size=10,droprates=0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(image_size[0], 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)

        self.image_size=image_size
        if self.image_size[1]==28:
            self.CNNoutputsize=3*3*64
        elif self.image_size[1]==32:
            self.CNNoutputsize=4*4*64


        self.fc1 = nn.Linear(self.CNNoutputsize, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

        self.droprates=droprates

        self.DIY_Dropout=DIY_Dropout(droprates)###make it part of the model so it gets the train/eval state


    def forward(self, x):

        #x=F.dropout(x, p=self.droprates[0], training=self.training)
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        x = x.view(-1,self.CNNoutputsize)
        x = F.relu(self.fc1(x))
        x=self.DIY_Dropout(x)
        #x = F.dropout(x, p=self.droprates[1],training=self.training)
        x = self.fc2(x)
        return x

class CNN_MaskedDropout(nn.Module):
    def __init__(self,image_size,hidden_size=10,droprates=0):
        super(CNN_MaskedDropout, self).__init__()
        self.conv1 = nn.Conv2d(image_size[0], 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)

        self.image_size=image_size
        if self.image_size[1]==28:
            self.CNNoutputsize=3*3*64
        elif self.image_size[1]==32:
            self.CNNoutputsize=4*4*64

        self.fc1 = nn.Linear(self.CNNoutputsize, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

        self.droprates=droprates

        self.Mask_Dropout=Mask_Dropout()###make it part of the model so it gets the train/eval state


    def forward(self, x,mask):

        #x=F.dropout(x, p=self.droprates[0], training=self.training)
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        x = x.view(-1,self.CNNoutputsize)
        x = F.relu(self.fc1(x))
        x=self.Mask_Dropout(x,mask)
        #x = F.dropout(x, p=self.droprates[1],training=self.training)
        x = self.fc2(x)
        return x

    def Get_condition(self, x):

        #x=F.dropout(x, p=self.droprates[0], training=self.training)
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        x = x.view(-1,self.CNNoutputsize)
        x = F.relu(self.fc1(x))
     
        return x


class CNN_Standout(nn.Module):
    def __init__(self,image_size,hidden_size=10,droprates=0):
        super(CNN_Standout, self).__init__()
        self.conv1 = nn.Conv2d(image_size[0], 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)



        self.image_size=image_size
        if self.image_size[1]==28:
            self.CNNoutputsize=3*3*64
        elif self.image_size[1]==32:
            self.CNNoutputsize=4*4*64

        self.fc1 = nn.Linear(self.CNNoutputsize, hidden_size)


        self.fc1_drop = Standout(self.fc1, droprates, 1)

        self.fc2 = nn.Linear(hidden_size, 10)

        self.droprates=droprates

        

    def forward(self, x):

        #x=F.dropout(x, p=self.droprates[0], training=self.training)
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        x = x.view(-1,self.CNNoutputsize )
        
        previous = x
        x_relu = F.relu(self.fc1(x))
        # Select between dropouts styles
        x = self.fc1_drop(previous, x_relu)
        #x=self.DIY_Dropout(x)
        #x = F.dropout(x, p=self.droprates[1],training=self.training)
        x = self.fc2(x)
        return x

class CNN_SVD(nn.Module):
    def __init__(self,image_size,hidden_size=10,droprates=0,threshold=3):
        super(CNN_SVD, self).__init__()
        

        self.conv1 = nn.Conv2d(image_size[0], 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        #self.fc1 = nn.Linear(3*3*64, hidden_size)
        
        self.image_size=image_size
        if self.image_size[1]==28:
            self.CNNoutputsize=3*3*64
        elif self.image_size[1]==32:
            self.CNNoutputsize=4*4*64
        
        self.fc1 = LinearSVDO(self.CNNoutputsize, hidden_size, threshold) 
        
        self.fc2 = nn.Linear(hidden_size, 10)

        self.droprates=droprates

        self.DIY_Dropout=DIY_Dropout(droprates)###make it part of the model so it gets the train/eval state


    def forward(self, x):

        #x=F.dropout(x, p=self.droprates[0], training=self.training)
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)

        x = x.view(-1,self.CNNoutputsize )
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=self.droprates[1],training=self.training)
        x = self.fc2(x)
        return x


# ResNet with SVD, Like CNN_SVD()
class ResNet_SVD(nn.Module):
   
    def __init__(self,num_layers,img_channels=3,hidden_size=10,droprates=0,threshold=3):
        super(ResNet_SVD, self).__init__()
        assert num_layers in [18, 34, 50, 101, 152], "ResNet: Unknown architecture! Number of layers has to be 18, 34, 50, 101, or 152 "

        block = Block
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.CNNoutputsize = 512 * self.expansion
     
        self.fc1 = LinearSVDO(self.CNNoutputsize, hidden_size, threshold) 
        
        self.fc2 = nn.Linear(hidden_size, 10)

        self.droprates=droprates

        self.DIY_Dropout=DIY_Dropout(droprates)###make it part of the model so it gets the train/eval state        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x= self.fc2(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)




# ResNet with Standout, Like CNN_Standout()
class ResNet_Standout(nn.Module):
   
    def __init__(self,num_layers,img_channels=3,hidden_size=10,droprates=0 ):
        super(ResNet_Standout, self).__init__()
        assert num_layers in [18, 34, 50, 101, 152], f'Unknown architecture! Number of layers has to be 18, 34, 50, 101, or 152'
        block = Block
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * self.expansion, hidden_size)
      
        self.fc1_drop = Standout(self.fc1, droprates, 1)

        self.fc2 = nn.Linear(hidden_size, 10)

        self.droprates=droprates
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        import pdb; pdb.set_trace()

        previous = x
        x_relu = F.relu(self.fc1(x))

        x = self.fc1_drop(previous,x_relu)
        x= self.fc2(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)

# ResNet with MaskedDropout like CNN_MaskedDropout()
class ResNet_MaskedDropout(nn.Module):
   
    def __init__(self,num_layers,img_channels=3,hidden_size=10,droprates=0 ):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet_MaskedDropout, self).__init__()
        block = Block
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * self.expansion, hidden_size)
        self.fc2 = nn.Linear(hidden_size,10)

        self.droprates=droprates

        self.Mask_Dropout=Mask_Dropout()###make it part of the model so it gets the train/eval state

    def forward(self, x,mask):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x=self.Mask_Dropout(x,mask)
        x= self.fc2(x)
        return x

    def Get_condition(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        
        return x    

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)


# ResNet with DIY_Dropout, Like CNN()
class ResNet(nn.Module):
   
    def __init__(self,num_layers,img_channels=3,hidden_size=10,droprates=0,threshold=3 ):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        block = Block
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * self.expansion, hidden_size)
        self.fc2 = nn.Linear(hidden_size,10)

        self.droprates=droprates

        self.DIY_Dropout=DIY_Dropout(droprates)###make it part of the model so it gets the train/eval state

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
       
        x = self.fc1(x)
        x= self.DIY_Dropout(x)
        x= self.fc2(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)


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

   
    res = ResNet_Standout(18)
    y = res(torch.randn(4, 3, 224, 224))
    # print(y.size())
    # torch.Size([4, 10])