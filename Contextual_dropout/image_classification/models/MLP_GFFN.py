import torch
import torch.nn as nn

#from models.layer import ARMDense_GFFN as ARMDense
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

import torch.optim as optim

epsilon = 1e-7
class ARMMLP_GFFN(nn.Module):
    def __init__(self, input_dim=784, num_classes=10, N=60000, layer_dims=(300, 100), beta_ema=0.999,
                 weight_decay=5e-4, lambas=(.1, .1, .1), local_rep=True,opt=None):
        super(ARMMLP_GFFN, self).__init__()

        self.opt=opt
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.N = N
        self.weight_decay = N * weight_decay
        self.lambas = lambas
        self.epoch = 0
        self.elbo = torch.zeros(1)
        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
            droprate_init, lamba = 0.2 if i == 0 else self.opt.mlp_dr, lambas[i] if len(lambas) > 1 else lambas[0]
            layers +=[nn.Linear(inp_dim, dimh)]

        # layers.append(ARMDense(self.layer_dims[-1], num_classes, droprate_init=self.opt.mlp_dr,
        #                        weight_decay=self.weight_decay, lamba=lambas[-1],
        #                         local_rep=self.opt.local_rep,opt=self.opt))
        layers.append(nn.Linear(self.layer_dims[-1],num_classes))
        self.output = nn.Sequential(*layers)
        self.layers = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.layers.append(m)



        ###construct the mask for GFFN

        self.rand_mask_generators=construct_random_mask_generators(layers=self.layers[:(len(self.layers)-1)],
                                                                    dropout_rate=opt.dropout_rate)
        self.mask_generators=construct_mlp_mask_generators(layers=self.layers[:(len(self.layers)-1)],
                                                        dropout_rate=opt.dropout_rate)
        self.activation=nn.ReLU
        mg_activation=nn.LeakyReLU
        z_lr=1e-1
        mg_lr=1e-3
        self.total_flowestimator = MLP_GFFN(in_dim=input_dim,out_dim=1,
                        activation=mg_activation)

        MaskGeneratorParameters=[]
        for generator in self.mask_generators:
            MaskGeneratorParameters+=list(generator.parameters())
       
        param_list = [{'params': MaskGeneratorParameters, 'lr': mg_lr},
                     {'params': self.total_flowestimator.parameters(), 'lr': z_lr}]
        

        self.mg_optimizer = optim.Adam(param_list)


    def score(self, x):
        return self.output(x.view(-1, self.input_dim))

    def forward(self, x, y=None):


        #using GFlownet

        x=x.view(-1, self.input_dim)
        
        if self.training:
            score,_=self.GFFN_forward(x)

        else:

            N_repeats=5#sample multiple times and use average as inference prediciton because GFN cannot take expection easily
            score=[]
            for _ in range(N_repeats):
                score_,_ = self.GFFN_forward(x)
                score.append(score_.unsqueeze(2))
            score=torch.cat(score,2).mean(2)   
            
        return score



    def regularization(self):
        regularization = 0.
        # for layer in self.layers:
        #     regularization += (1. / self.N) * layer.regularization()
        return regularization

    def prune_rate(self):
        return 0.0



    #####GFFFN related functions
    def GFFN_forward(self, x):
        masks=[]
        for layer, mg in zip(self.layers[:(len(self.layers)-1)],self.mask_generators):
            ##dropingout the inputs and all the hidden layers but not the output layer
            x=layer(x)
            x = self.activation()(x)
            # generate mask & dropout
          
            m = mg(x).detach()

            masks.append(m)
            multipliers = m.shape[1] / (m.sum(1) + 1e-6)
            x = torch.mul((x * m).T, multipliers).T
        score=self.layers[-1](x)
        return score, masks

    def GFFN_forward_predefinedmasks(self, x,predefined_masks):
        masks=[]
        for layer, m in zip(self.layers[:(len(self.layers)-1)],predefined_masks):
            ##dropingout the inputs and all the hidden layers but not the output layer
            x=layer(x)
            x = self.activation()(x)
            
            m = m.detach()
            masks.append(m)
            multipliers = m.shape[1] / (m.sum(1) + 1e-6)
            x = torch.mul((x * m).T, multipliers).T
        score=self.layers[-1](x)
        return score, masks

    def _gfn_step(self, x_mask, y_mask,x_reward=None,y_reward=None):
        #####this step allows us to use different x,y to generate mask and calcualte reward(loss)

        metric = {}
        x_mask=x_mask.view(-1, self.input_dim)
        if x_reward!=None:
            #generate mask
            x_reward=x_reward.view(-1, self.input_dim)
            _, masks = self.GFFN_forward(x_mask)

            ###for loss
            logits, _ = self.GFFN_forward_predefinedmasks(x_reward, masks)
        
        else:
            logits, masks = self.GFFN_forward(x_mask)
            x_reward=x_mask
            y_reward=y_mask 


        beta=1#temperature
        with torch.no_grad():
            losses = nn.CrossEntropyLoss(reduce=False)(logits, y_reward)
            log_rewards = - beta * losses
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

#####code related to GFFN mask generating

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
        # if self.dropout_rate!=None:
        #     dist = (1. - self.dropout_rate) * self.num_unit * x / (x.sum(1).unsqueeze(1) + 1e-6)
        #     dist = dist.clamp(0, 1)
        # else:
        dist=x

        return dist

    def forward(self, x):
        
     
        return torch.bernoulli(self._dist(x))

    def log_prob(self, x, m):
        dist = self._dist(x)
        probs = dist * m + (1. - dist) * (1. - m)
        return torch.log(probs).sum(1)


def construct_random_mask_generators(
        layers,
        dropout_rate,
):
    mask_generators = nn.ModuleList()
    for layer in layers:
        mask_generators.append(
            RandomMaskGenerator(
                dropout_rate=dropout_rate,
            )
        )

    return mask_generators


def construct_mlp_mask_generators(
        layers,
        dropout_rate,
        hidden=None,
        activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()
    for layer in layers:
        mask_generators.append(
            MLPMaskGenerator(
                num_unit=layer.weight.shape[0],
                dropout_rate=dropout_rate,
                hidden=hidden,
                activation=activation
            )
        )

    return mask_generators



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