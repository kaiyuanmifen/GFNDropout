
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

  
        self.fc1 = nn.Linear(Input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
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






################GFFN (faster version)

class MLP_GFFN(nn.Module):
    def __init__(self, in_dim=784, out_dim=10, hidden=None, activation=nn.LeakyReLU):
        super().__init__()
        if hidden is None:
            hidden = [32, 32]

        self.LN=nn.LayerNorm(in_dim)##normalize the input to prevent exploding
        h_old = in_dim
        self.fc = nn.ModuleList()
        for h in hidden:
            self.fc.append(nn.Linear(h_old, h))
            h_old = h
        self.out_layer = nn.Linear(h_old, out_dim)
        self.activation = activation

    def forward(self, x):
        x=self.LN(x)
        for layer in self.fc:
            x = self.activation()(layer(x))
        x = self.out_layer(x)
        return x


class MLPMaskedDropout(nn.Module):

    def __init__(self, in_dim=784, out_dim=10, hidden=None, activation=nn.LeakyReLU):
        super().__init__()
        if hidden is None:
            hidden = [32, 32]
        h_old = in_dim
        self.fc = nn.ModuleList()
        for h in hidden:
            self.fc.append(nn.Linear(h_old, h))
            h_old = h
        self.out_layer = nn.Linear(h_old, out_dim)
        self.activation = activation

    def forward(self, x, mask_generators):
        masks = []
        for layer, mg in zip(self.fc, mask_generators):
            x = self.activation()(layer(x))
            # generate mask & dropout

            m = mg(x).detach()
            masks.append(m)
            multipliers = m.shape[1] / (m.sum(1) + 1e-6)
            x = torch.mul((x * m).T, multipliers).T
        x = self.out_layer(x)
        return x, masks

    def forward_predifinedMask(self, x, mask):
        ###forward using predefined mask
        masks = []
        for layer, m in zip(self.fc, mask):
            x = self.activation()(layer(x))
            
            masks.append(m)
            multipliers = m.shape[1] / (m.sum(1) + 1e-6)
            x = torch.mul((x * m).T, multipliers).T
        x = self.out_layer(x)
        return x, masks


class RandomMaskGenerator(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, x):
        return torch.bernoulli((1. - self.dropout_rate) * torch.ones(x.shape))

    def log_prob(self, x, m):
        dist = (1. - self.dropout_rate) * torch.ones(x.shape).to(self.device)
        probs = dist * m + (1. - dist) * (1. - m)
        return torch.log(probs).sum(1)


class MLPMaskGenerator(nn.Module):
    def __init__(self, num_unit, dropout_rate, hidden=None, activation=nn.LeakyReLU):
        super().__init__()
        self.num_unit = torch.tensor(num_unit).type(torch.float32)
        self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)
        self.mlp = MLP_GFFN(
            in_dim=num_unit,
            out_dim=num_unit,
            hidden=hidden,
            activation=activation,
        )

    def _dist(self, x):
        x = self.mlp(x)

        x = torch.sigmoid(x)
    
        # dist = (1. - self.dropout_rate) * self.num_unit * x / (x.sum(1).unsqueeze(1) + 1e-6)
        #dist = dist.clamp(0, 1)
        dist=x

        return dist

    def forward(self, x):
        
     
        return torch.bernoulli(self._dist(x))

    def log_prob(self, x, m):
        dist = self._dist(x)
        probs = dist * m + (1. - dist) * (1. - m)
        return torch.log(probs).sum(1)


def construct_random_mask_generators(
        model,
        dropout_rate,
):
    mask_generators = nn.ModuleList()
    for layer in model.fc:
        mask_generators.append(
            RandomMaskGenerator(
                dropout_rate=dropout_rate,
            )
        )

    return mask_generators


def construct_mlp_mask_generators(
        model,
        dropout_rate,
        hidden=None,
        activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()
    for layer in model.fc:
        mask_generators.append(
            MLPMaskGenerator(
                num_unit=layer.weight.shape[0],
                dropout_rate=dropout_rate,
                hidden=hidden,
                activation=activation
            )
        )

    return mask_generators


class MLPClassifierWithMaskGenerator(nn.Module):
    def __init__(
            self,
            in_dim=784,
            out_dim=10,
            hidden=None,
            activation=nn.LeakyReLU,
            dropout_rate=0.5,
            mg_type='random',
            lr=1e-3,
            z_lr=1e-1,
            mg_lr=1e-3,
            mg_hidden=None,
            mg_activation=nn.LeakyReLU,
            beta=0.1,
            device='cpu',
    ):
        super().__init__()
        # classifier
        self.model = MLPMaskedDropout(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            activation=activation
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # mask generators
        self.mg_type = mg_type
        if mg_type == 'random':
            self.mask_generators = construct_random_mask_generators(
                model=self.model,
                dropout_rate=dropout_rate
            ).to(device)
        elif mg_type == 'gfn':
            # for backward log prob calculation only
            self.rand_mask_generators = construct_random_mask_generators(
                model=self.model,
                dropout_rate=dropout_rate
            ).to(device)
            self.mask_generators = construct_mlp_mask_generators(
                model=self.model,
                dropout_rate=dropout_rate,
                hidden=mg_hidden,
                activation=mg_activation,
            ).to(device)
            
            ####total flow (logZ) should condition on input or previous layoutout
            #self.logZ = nn.Parameter(torch.tensor(0.)).to(device)
            
            self.total_flowestimator = MLP_GFFN(in_dim=in_dim,out_dim=1,
                                    activation=mg_activation).to(device)
            
                        
            #param_list = [{'params': self.model.parameters(), 'lr': mg_lr},
            #              {'params': self.logZ, 'lr': z_lr}]

            MaskGeneratorParameters=[]
            for generator in self.mask_generators:
                MaskGeneratorParameters+=list(generator.parameters())
           
            param_list = [{'params': MaskGeneratorParameters, 'lr': mg_lr},
                         {'params': self.total_flowestimator.parameters(), 'lr': z_lr}]
            

            self.mg_optimizer = optim.Adam(param_list)
        else:
            raise ValueError('unknown mask generator type {}'.format(mg_type))

        # gfn parameters
        self.beta = beta

    def step(self, x, y):
        metric = {}
        logits, masks = self.model(x, self.mask_generators)
        # Update model
        self.optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
        metric['loss'] = loss.item()
        metric['acc'] = acc
        loss.backward()
        self.optimizer.step()

        # # Update mask generators
        # if self.mg_type == 'gfn':
        #     if x_valid is not None and y_valid is not None:
        #         #update using validation set
        #         metric.update(self._gfn_step(x_valid, y_valid,x_valid, y_valid))
        #     else:
        #         metric.update(self._gfn_step(x, y,x,y))


        return logits,metric

    def _gfn_step(self, x_mask, y_mask,x_reward,y_reward):
        #####this step allows us to use different x,y to generate mask and calcualte reward(loss)

        metric = {}
        #generate mask
        _, masks = self.model(x_mask, self.mask_generators)

        ###for loss
        logits, _ = self.model.forward_predifinedMask(x_reward, masks)
                

        with torch.no_grad():
            losses = nn.CrossEntropyLoss(reduce=False)(logits, y_reward)
            log_rewards = - self.beta * losses
            logZ=self.total_flowestimator(x_mask)#this flow is calculated using x_mask, not a bug , to encourage generalization 
        # trajectory balance loss
        log_probs_F = []
        log_probs_B = []
        for m, mg_f, mg_b in zip(masks, self.mask_generators, self.rand_mask_generators):
            log_probs_F.append(mg_f.log_prob(m, m).unsqueeze(1))
            log_probs_B.append(mg_b.log_prob(m, m).unsqueeze(1))
        tb_loss = ((logZ - log_rewards
                    + torch.cat(log_probs_F, dim=1).sum(dim=1)
                    - torch.cat(log_probs_B, dim=1).sum(dim=1)) ** 2).mean()
        metric['tb_loss'] = tb_loss.item()
        self.mg_optimizer.zero_grad()
        tb_loss.backward()
        self.mg_optimizer.step()


        ###calculated actual droppout rate
        actual_dropout_rate=0
        for m in masks:
            actual_dropout_rate+=m.sum(1).mean(0)/(m.shape[1]) 
        metric['actual_dropout_rate']=(actual_dropout_rate/len(masks)).item()



        return metric

    def test(self, x, y):
        metric = {}
        logits, masks = self.model(x, self.mask_generators)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
        metric['loss'] = loss.item()
        metric['acc'] = acc

        return metric

    def forward(self,x):
        logits, masks = self.model(x, self.mask_generators)

        return logits


########baselines for GFFN 

class MLP_Alldrop(nn.Module):
    def __init__(self,Input_size, hidden_size=10, droprates=0):
        super(MLP_Alldrop, self).__init__()
        ###this version has dropout on on layers

        self.fc1 = nn.Linear(Input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.output = nn.Linear(hidden_size, 10)


        self.DIY_Dropout=DIY_Dropout(droprates)###make it part of the model so it gets the train/eval state

        self.droprates=droprates

    def forward(self, x):
      
        x = x.view(x.shape[0], -1)
        #x=self.DIY_Dropout(x)
        x=F.relu(self.fc1(x))
        x=self.DIY_Dropout(x)
        x=F.relu(self.fc2(x))
        x=self.DIY_Dropout(x)
        x=F.relu(self.fc3(x))
        x=self.DIY_Dropout(x)
        #x=DIY_Dropout(p=self.droprates[1])(x)
        x=self.output(x)
        return x




class MLP_SVDAll(nn.Module):
    ####SVD on all layers
    def __init__(self, Input_size,hidden_size=10,threshold=3):
        super(MLP_SVDAll, self).__init__()

        #self.fc1 = nn.Linear(28*28,hidden_size)
        self.fc1 = LinearSVDO(Input_size, hidden_size, threshold) 
        self.fc2 = LinearSVDO(hidden_size, hidden_size, threshold)
        self.fc3 = LinearSVDO(hidden_size, hidden_size, threshold)
        self.output = nn.Linear(hidden_size, 10)

    def forward(self, x):
      
        x = x.view(x.shape[0], -1)
        #x=DIY_Dropout(p=self.droprates[0])(x)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        #x=DIY_Dropout(p=self.droprates[1])(x)
        x=self.output(x)
        return x




class MLP_StandoutAll(nn.Module):
    def __init__(self,Input_size, hidden_size=10,droprates=0.5):
        super(MLP_StandoutAll, self).__init__()

        self.fc1 = nn.Linear(Input_size,hidden_size)
   
        self.fc1_drop = Standout(self.fc1, droprates, 1)     
        
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        
        self.fc2_drop = Standout(self.fc2, droprates, 1)

        self.fc3 = nn.Linear(hidden_size,hidden_size)
        
        self.fc3_drop = Standout(self.fc3, droprates, 1)
        
 
        
        self.output = nn.Linear(hidden_size, 10)

    def forward(self, x):
      
        x = x.view(x.shape[0], -1)
        #x=DIY_Dropout(p=self.droprates[0])(x)
        
        previous = x        
        x_relu= F.relu(self.fc1(x))
        x = self.fc1_drop(previous, x_relu)
       

        previous = x
        x_relu = F.relu(self.fc2(x))
        x = self.fc2_drop(previous, x_relu)
       
        previous = x
        x_relu = F.relu(self.fc3(x))
        x = self.fc3_drop(previous, x_relu)

              
        x=self.output(x)

        return x




#####resenet version

import torchvision.models as models
from collections import OrderedDict 

class Resenet_Alldrop(nn.Module):
    def __init__(self,num_classes,image_size,hidden_size,droprates):
        super(Resenet_Alldrop, self).__init__()

        self.image_size=image_size
        self.resnet18 = models.resnet18(pretrained=False)
        #print(self.resnet18)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, hidden_size)

        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.output = nn.Linear(hidden_size, 10)


        self.DIY_Dropout=DIY_Dropout(droprates)###make it part of the model so it gets the train/eval state

        self.droprates=droprates
        
    def forward(self, x):
        ###x come in the size (bsz,-1)

        ###put x back into image size
        x=x.reshape(x.shape[0],self.image_size[0],self.image_size[1],self.image_size[2])

        x=self.resnet18(x)

        x=self.DIY_Dropout(x)
        x=F.relu(self.fc2(x))
        x=self.DIY_Dropout(x)
        x=F.relu(self.fc3(x))
        x=self.DIY_Dropout(x)
        #x=DIY_Dropout(p=self.droprates[1])(x)
        x=self.output(x)

        return x




class Resenet_GFFN(nn.Module):
    def __init__(
            self,
            image_size,
            in_dim=784,
            out_dim=10,
            hidden=None,
            activation=nn.LeakyReLU,
            dropout_rate=0.5,
            mg_type='random',
            lr=1e-3,
            z_lr=1e-1,
            mg_lr=1e-3,
            mg_hidden=None,
            mg_activation=nn.LeakyReLU,
            beta=0.1,
            device='cpu',
    ):
        super().__init__()

        ####Resnet

        self.image_size=image_size
        self.resnet18 = models.resnet18(pretrained=False).to(device)
        #print(self.resnet18)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, in_dim)

        # classifier
        self.model = MLPMaskedDropout(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            activation=activation
        ).to(device)
        self.optimizer = optim.Adam(list(self.model.parameters())+list(self.resnet18.parameters()), lr=lr)

        # mask generators
        self.mg_type = mg_type
        if mg_type == 'random':
            self.mask_generators = construct_random_mask_generators(
                model=self.model,
                dropout_rate=dropout_rate
            ).to(device)
        elif mg_type == 'gfn':
            # for backward log prob calculation only
            self.rand_mask_generators = construct_random_mask_generators(
                model=self.model,
                dropout_rate=dropout_rate
            ).to(device)
            self.mask_generators = construct_mlp_mask_generators(
                model=self.model,
                dropout_rate=dropout_rate,
                hidden=mg_hidden,
                activation=mg_activation,
            ).to(device)
            
            ####total flow (logZ) should condition on input or previous layoutout
            #self.logZ = nn.Parameter(torch.tensor(0.)).to(device)
            
            self.total_flowestimator = MLP_GFFN(in_dim=in_dim,out_dim=1,
                                    activation=mg_activation).to(device)
            
                        
            #param_list = [{'params': self.model.parameters(), 'lr': mg_lr},
            #              {'params': self.logZ, 'lr': z_lr}]

            MaskGeneratorParameters=[]
            for generator in self.mask_generators:
                MaskGeneratorParameters+=list(generator.parameters())
           
            param_list = [{'params': MaskGeneratorParameters, 'lr': mg_lr},
                         {'params': self.total_flowestimator.parameters(), 'lr': z_lr}]
            

            self.mg_optimizer = optim.Adam(param_list)
        else:
            raise ValueError('unknown mask generator type {}'.format(mg_type))

        # gfn parameters
        self.beta = beta

    def step(self, x, y):

        ###x come in the size (bsz,-1)

        ###put x back into image size
        x=x.reshape(x.shape[0],self.image_size[0],self.image_size[1],self.image_size[2])

        ###go through resnet first
        x=self.resnet18(x)

        ###then get into the FFL layers
        metric = {}
        logits, masks = self.model(x, self.mask_generators)
        # Update model
        self.optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
        metric['loss'] = loss.item()
        metric['acc'] = acc
        loss.backward()
        self.optimizer.step()

        # # Update mask generators
        # if self.mg_type == 'gfn':
        #     if x_valid is not None and y_valid is not None:
        #         #update using validation set
        #         metric.update(self._gfn_step(x_valid, y_valid,x_valid, y_valid))
        #     else:
        #         metric.update(self._gfn_step(x, y,x,y))

        return logits,metric

    def _gfn_step(self, x_mask, y_mask,x_reward,y_reward):

        ###x come in the size (bsz,-1)

        ###put x back into image size
        x_mask= x_mask.reshape(x_mask.shape[0],self.image_size[0],self.image_size[1],self.image_size[2])
        x_reward= x_reward.reshape(x_reward.shape[0],self.image_size[0],self.image_size[1],self.image_size[2])


        ###go through resnet first
        x_mask=self.resnet18(x_mask)
        x_reward=self.resnet18(x_reward)
                

        #####this step allows us to use different x,y to generate mask and calcualte reward(loss)

        metric = {}
        #generate mask
        _, masks = self.model(x_mask, self.mask_generators)

        ###for loss
        logits, _ = self.model.forward_predifinedMask(x_reward, masks)
                

        with torch.no_grad():
            losses = nn.CrossEntropyLoss(reduce=False)(logits, y_reward)
            log_rewards = - self.beta * losses
            logZ=self.total_flowestimator(x_mask)#this flow is calculated using x_mask, not a bug , to encourage generalization 
        # trajectory balance loss
        log_probs_F = []
        log_probs_B = []
        for m, mg_f, mg_b in zip(masks, self.mask_generators, self.rand_mask_generators):
            log_probs_F.append(mg_f.log_prob(m, m).unsqueeze(1))
            log_probs_B.append(mg_b.log_prob(m, m).unsqueeze(1))
        tb_loss = ((logZ - log_rewards
                    + torch.cat(log_probs_F, dim=1).sum(dim=1)
                    - torch.cat(log_probs_B, dim=1).sum(dim=1)) ** 2).mean()
        metric['tb_loss'] = tb_loss.item()
        self.mg_optimizer.zero_grad()
        tb_loss.backward()
        self.mg_optimizer.step()

        return metric

    def test(self, x, y):

        ###x come in the size (bsz,-1)

        ###put x back into image size
        x=x.reshape(x.shape[0],self.image_size[0],self.image_size[1],self.image_size[2])

        ###go through resnet first
        x=self.resnet18(x)

        metric = {}
        logits, masks = self.model(x, self.mask_generators)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
        metric['loss'] = loss.item()
        metric['acc'] = acc

        return metric

    def forward(self,x):

        ###x come in the size (bsz,-1)

        ###put x back into image size
        x=x.reshape(x.shape[0],self.image_size[0],self.image_size[1],self.image_size[2])

        ###go through resnet first
        x=self.resnet18(x)

        logits, masks = self.model(x, self.mask_generators)

        return logits


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