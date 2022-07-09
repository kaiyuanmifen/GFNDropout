# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backbone subnetwork based on ResNet-20 and ResNet-56."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_weights
import torch.optim as optim

class BasicBlock(nn.Module):
  """Basic block of ResNet with filters gated."""

  def __init__(self, in_channels, out_channels, stride):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)
    self.conv2 = nn.Conv2d(
        out_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()
    if in_channels != out_channels:
      self.shortcut.add_module(
          'conv',
          nn.Conv2d(
              in_channels,
              out_channels,
              kernel_size=1,
              stride=stride,
              padding=0,
              bias=False))
      self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

  def forward(self, x):
    
    '''
    x has shape [bsz,n_chanel, H,W]
    '''


    y = F.relu(self.bn1(self.conv1(x)), inplace=True)
    y = self.bn2(self.conv2(y))

    y += self.shortcut(x)
    y = F.relu(y, inplace=True)
    return y


class Network(nn.Module):
  """Backbone network based on ResNet."""

  def __init__(self, args,depth=20, num_classes=10,):
    super(Network, self).__init__()
    input_shape = [1, 3, 32, 32]

    base_channels = 16
    num_blocks_per_stage = (depth - 2) // 6

    n_channels = [base_channels, base_channels * 2, base_channels * 4]
    
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    self.conv = nn.Conv2d(
        input_shape[1],
        n_channels[0],
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False)
    self.bn = nn.BatchNorm2d(base_channels)


    self.ResNetblocks=nn.ModuleList()
    ###make different stages of ResNet and append it to the block list
    self._make_stage(
        n_channels[0], n_channels[0],
        num_blocks_per_stage, stride=1)
    self._make_stage(
        n_channels[0], n_channels[1],
        num_blocks_per_stage, stride=2)
    self._make_stage(
        n_channels[1], n_channels[2],
        num_blocks_per_stage, stride=2)
   
   # print('Total number of gates needed: {}\n'.format(self.gate_size))




    #####mask generator 
    if args.method=="GFFN":
      self.mask_generators = construct_mask_generators(
      n_channels=[n_channels[0]]*num_blocks_per_stage+[n_channels[1]]*num_blocks_per_stage+[n_channels[2]]*num_blocks_per_stage,
      dropout_rate=args.dropout_rate,
      hidden=[32,32],
      activation=nn.LeakyReLU,)

    elif args.method=="StandardDropout":
      #use random dropout
      self.mask_generators = construct_random_mask_generators(    
      n_channels=[n_channels[0]]*num_blocks_per_stage+[n_channels[1]]*num_blocks_per_stage+[n_channels[2]]*num_blocks_per_stage,
      dropout_rate=args.dropout_rate,)

    ###random mask generator
    self.rand_mask_generators=construct_random_mask_generators(    
    n_channels=[n_channels[0]]*num_blocks_per_stage+[n_channels[1]]*num_blocks_per_stage+[n_channels[2]]*num_blocks_per_stage,
    dropout_rate=args.dropout_rate,)




    # Get feature size.
    with torch.no_grad():
      self.feature_size = self._get_conv_features(
          x=torch.ones(*input_shape), mask_generators=self.mask_generators)[0].view(-1).shape[0]


    self.fc = nn.Linear(self.feature_size, num_classes)


    ###GFN flow estimator
    self.total_flowestimator = MLP_GFFN(in_dim=3*32*32,out_dim=1,
                        activation=nn.LeakyReLU)

    # gfn parameters
    self.beta = args.beta



    z_lr=1e-1
    mg_lr=1e-3

    GFN_param_list = [{'params': self.mask_generators.parameters(), 'lr': mg_lr},
             {'params': list(self.total_flowestimator.parameters()), 'lr': z_lr}]

    self.mg_optimizer = optim.Adam(GFN_param_list)


    # Initialize weights.
    self.apply(init_weights)

  def _make_stage(self, in_channels, out_channels, num_blocks,
                  stride):

    for i in range(num_blocks):
      name = 'block{}'.format(i + 1)
      if i == 0:
        self.ResNetblocks.append(BasicBlock(in_channels,
                                out_channels,
                                stride=stride))
      else:
        self.ResNetblocks.append(
                     BasicBlock(out_channels,
                                out_channels,
                                stride=1))


  def _get_conv_features(self, x, mask_generators,provided_masks=None):
    y = F.relu(self.bn(self.conv(x)), inplace=True)

    if provided_masks==None:
      
      generated_masks=[]
      #generate mask using previous layer's activation
      

      for layerIdxin in range(len(self.ResNetblocks)):
        y= self.ResNetblocks[layerIdxin](y)

        m=self.mask_generators[layerIdxin](y.mean(3).mean(2))#mean across H and W dimension to simplify computation
        
        y = y * m.unsqueeze(dim=2).unsqueeze(dim=3)
        
        generated_masks.append(m)


      y = F.adaptive_avg_pool2d(y, output_size=1)

      return y,generated_masks

    else:
      #using provided maskes
      for layerIdxin in range(len(self.ResNetblocks)):
        y= self.ResNetblocks[layerIdxin](y)

        m=provided_masks[layerIdxin]
        y = y * m.unsqueeze(dim=2).unsqueeze(dim=3)


      y = F.adaptive_avg_pool2d(y, output_size=1)

  
      return y,provided_masks

  def forward(self, x):
    y,_ = self._get_conv_features(x, self.mask_generators)
    y = y.view(y.size(0), -1)
    y = self.fc(y)
    return y

  def _gfn_step(self, x_mask, y_mask,x_reward=None,y_reward=None):
      #####this step allows us to use different x,y to generate mask and calcualte reward(loss)

      metric = {}


      #generate mask
      pred, masks = self._get_conv_features(x_mask, self.mask_generators)

      target=y_mask
      if x_reward!=None:
        ##different x and y can be used for loss/reward ,eg validation
        pred,_ = self._get_conv_features(x_reward, self.mask_generators)
        target=y_reward
      pred = pred.view(pred.size(0), -1)
      logits = self.fc(pred)



      with torch.no_grad():
          losses = nn.CrossEntropyLoss(reduce=False)(logits, target)
          log_rewards = - self.beta * losses
          logZ=self.total_flowestimator(x_mask.view(x_mask.shape[0],-1))#this flow is calculated using x_mask, not a bug , to encourage generalization 
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


      return metric['tb_loss'] 




#################GFFN related codes


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
        x=x.view(x.shape[0],-1)
        x=self.LN(x)
        for layer in self.fc:
            x = self.activation()(layer(x))
        x = self.out_layer(x)
        return x




class RandomMaskGenerator(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, x):
        return torch.bernoulli((1. - self.dropout_rate) * torch.ones(x.shape)).to(self.device)

    def log_prob(self, x, m):
        dist = (1. - self.dropout_rate) * torch.ones(x.shape).to(self.device)
        probs = dist * m + (1. - dist) * (1. - m)
        return torch.log(probs).sum(1)


class MaskGenerator(nn.Module):
    def __init__(self, input_shape,num_unit, dropout_rate, hidden=None, activation=nn.LeakyReLU):
        super().__init__()
        self.num_unit = torch.tensor(num_unit).type(torch.float32)
        self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)
        self.mlp = MLP_GFFN(
            in_dim=input_shape,
            out_dim=num_unit,
            hidden=hidden,
            activation=activation,
        )

    def _dist(self, x):
        x = self.mlp(x)
 
        x = torch.sigmoid(x)
        dist = (1. - self.dropout_rate) * self.num_unit * x / (x.sum(1).unsqueeze(1) + 1e-6)
        dist = dist.clamp(0, 1)
        return dist

    def forward(self, x):
        
     
        return torch.bernoulli(self._dist(x))

    def log_prob(self, x, m):
        dist = self._dist(x)
        probs = dist * m + (1. - dist) * (1. - m)
        return torch.log(probs).sum(1)


def construct_random_mask_generators(
        n_channels,
        dropout_rate,
):
    mask_generators = nn.ModuleList()
    for n_c in n_channels:
        mask_generators.append(
            RandomMaskGenerator(
                dropout_rate=dropout_rate,
            )
        )

    return mask_generators



def construct_mask_generators(
        n_channels,
        dropout_rate,
        hidden=None,
        activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()
    for n_c in n_channels:
        mask_generators.append(
            MaskGenerator(
                input_shape=n_c,
                num_unit=n_c,
                dropout_rate=dropout_rate,
                hidden=hidden,
                activation=activation
            )
        )

    return mask_generators