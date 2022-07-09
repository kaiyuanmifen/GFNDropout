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

"""Loads a GaterNet checkpoint and tests on Cifar-10 test set."""

import argparse
import io
import os
from backbone_resnet import Network as Backbone
from backbone_resnet_GFFN import Network as Backbone_GFFN
from gater_resnet import Gater
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import pandas as pd
import time

def load_from_state(state_dict, model):
  """Loads the state dict of a checkpoint into model."""
  tem_dict = dict()
  for k in state_dict.keys():
    tem_dict[k.replace('module.', '')] = state_dict[k]
  state_dict = tem_dict

  ckpt_key = set(state_dict.keys())
  model_key = set(model.state_dict().keys())
  print('Keys not in current model: {}\n'.format(ckpt_key - model_key))
  print('Keys not in checkpoint: {}\n'.format(model_key - ckpt_key))

  model.load_state_dict(state_dict, strict=True)
  print('Successfully reload from state.')
  return model

def test(backbone, gater, device, test_loader,args):
  """Tests the model on a test set."""
  backbone.eval()
  if gater!=None:
    gater.eval()
  loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)

      if gater!=None:
        gate = gater(data)
        output = backbone(data, gate)
      else:
        output = backbone(data)
      loss += F.cross_entropy(output, target, size_average=False).item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()

  loss /= len(test_loader.dataset)
  acy = 100. * correct / len(test_loader.dataset)
  # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
  #     loss, correct, len(test_loader.dataset), acy))

  return acy

def train(backbone, gater, device, train_loader,optimizer,args):
  """train the model"""
  backbone.train()
  if gater!=None:
    gater.train()
  loss = 0
  correct = 0
  GFN_loss=0

  for data, target in train_loader:
    #print("batch")
    data, target = data.to(device), target.to(device)
    

    if gater!=None:
      gate = gater(data)
      output = backbone(data, gate)
    else:
      output = backbone(data)
    batch_loss=F.cross_entropy(output, target, size_average=False)
  
    loss += batch_loss.item()
    pred = output.max(1, keepdim=True)[1]
    correct += pred.eq(target.view_as(pred)).sum().item()

    ###optimization 
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    if args.method=="GFFN":
      #UDPATE GFN related parameters
      batch_GFN_loss=backbone._gfn_step(data, target)
      GFN_loss += batch_GFN_loss
  loss /= len(train_loader.dataset)
  acy = 100. * correct / len(train_loader.dataset)
  print('\ntrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%) GFN Loss  {:.4f}\n'.format(
      loss, correct, len(train_loader.dataset), acy,GFN_loss))

  return loss,acy,GFN_loss


def run(args, device, train_loader,valid_loader,test_loader):
  """Loads checkpoint into GaterNet and runs test on the test data."""
  # with open(args.checkpoint_file, 'rb') as fin:
  #   inbuffer = io.BytesIO(fin.read())
  # state_dict = torch.load(inbuffer, map_location='cpu')
  # print('Successfully load checkpoint file.\n')

  if args.method=="GaterNet":
    print("using Gaternet")
    backbone = Backbone(depth=args.backbone_depth, num_classes=10).to(device)
  elif args.method=="GFFN":
    print("Using GFFN")
    backbone=Backbone_GFFN(args,depth=args.backbone_depth, num_classes=10).to(device)
  
  elif args.method=="StandardDropout":
      print("Using StandardDropout")
      backbone=Backbone_GFFN(args,depth=args.backbone_depth, num_classes=10).to(device)


  #print('Loading checkpoint weights into backbone.')
  #backbone = load_from_state(state_dict['backbone_state_dict'], backbone)
  #backbone = nn.DataParallel(backbone).to(device)
  #print('Backbone is ready after loading checkpoint and moving to device:')
  print("backbone")
  print(backbone)
  n_params_b = sum(
      [param.view(-1).size()[0] for param in backbone.parameters()])
  print('Number of parameters in backbone: {}\n'.format(n_params_b))



  if args.method=="GaterNet":

    print("backbone.gate_size")
    print(backbone.gate_size)

    gater = Gater(depth=args.backbone_depth,
                  bottleneck_size=8,
                  gate_size=backbone.gate_size).to(device)
    #print('Loading checkpoint weights into gater.')
    #gater = load_from_state(state_dict['gater_state_dict'], gater)
   # gater = nn.DataParallel(gater).to(device)
    #print('Gater is ready after loading checkpoint and moving to device:')
    print("gater")
    print(gater)
    n_params_g = sum(
        [param.view(-1).size()[0] for param in gater.parameters()])
    print('Number of parameters in gater: {}'.format(n_params_g))
    print('Total number of parameters: {}\n'.format(n_params_b + n_params_g))
  else:
    gater= None





  #####optimizer

  #optimizer = optim.Adam(list(gater.parameters())+list(backbone.parameters()), lr=1e-3)
  if args.method=="GaterNet":
    optimizer = optim.Adam(list(gater.parameters())+list(backbone.parameters()), lr=0.05)
  else:
    optimizer = optim.Adam(list(backbone.parameters()), lr=0.05)

  milestones = [25, 40]

  scheduler = torch.optim.lr_scheduler.MultiStepLR(
      optimizer, milestones=milestones, gamma=0.1
  )

  print('Running train on train data.')
  best_acc=0


  all_train_loss,all_train_accs,all_GFN_loss,all_val_accs,all_test_accs=[],[],[],[],[]
  for epoch in range(1,args.n_epochs+1):
    print("epoch",epoch)
    start = time.time()
    train_loss,train_acc,GFN_loss=train(backbone, gater, device, train_loader,optimizer,args)
    val_acc=test(backbone, gater, device, valid_loader,args)
    test_acc=test(backbone, gater, device, test_loader,args)


    print('\n train average loss: {:.4f},train accuracy: ({:.4f}%) GFN loss: {:.4f} val_acc: ({:.4f}%) test_acc: ({:.4f}%) \n'.format(
    train_loss,train_acc,GFN_loss,val_acc,test_acc))
    
    scheduler.step()
    
    end = time.time()
    print("epoch takes time:",end - start)
    ####save results
    all_train_accs.append(train_acc)
    all_train_loss.append(train_loss)
    all_GFN_loss.append(GFN_loss)
    all_val_accs.append(val_acc)
    all_test_accs.append(test_acc)

    df = pd.DataFrame({"epoch":list(range(epoch)),
      'train_loss':all_train_loss,
      'all_GFN_loss':all_GFN_loss,
      'train_acc':all_train_accs,
      'val_acc':all_val_accs,
      'test_acc':all_test_accs})

    df.to_csv("Results/"+args.name+"_performance.csv")

    if val_acc>best_acc:
      best_acc=val_acc
      torch.save(backbone.state_dict(), "checkpoints/"+args.name+'_backbone.pt')
      if args.method=="GaterNet":
        torch.save(gater.state_dict(), "checkpoints/"+args.name+'_gater.pt')





def parse_flags():
  """Parses input arguments."""
  parser = argparse.ArgumentParser(description='GaterNet')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--backbone-depth', type=int, default=20,
                      help='resnet depth of the backbone subnetwork')
  parser.add_argument('--checkpoint-file', type=str, default=None,
                      help='checkpoint file to run test')
  parser.add_argument('--data-dir', type=str, default=None,
                      help='the directory for storing data')
  parser.add_argument('--n_epochs', type=int, default=800,
                    help='n_epochs')

  parser.add_argument('--name', type=str, default=None,
                help='name of the experiment')

  parser.add_argument('--method', type=str, default=None,
                help='method for drop out')

  parser.add_argument('--dropout_rate', type=float, default=0.5,
                help='rate of dropout')

  parser.add_argument('--beta', type=float, default=1.0,
                help='beta coefficient for the GFN reward')


  args = parser.parse_args()
  
  return args


def main(args):
  print('Input arguments:\n{}\n'.format(args))

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  print('use_cuda: {}'.format(use_cuda))

  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.backends.cudnn.benchmark = True
  print('device: {}'.format(device))

  if not os.path.isdir(args.data_dir):
    os.mkdir(args.data_dir)

  kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
  normalize_mean = [0.4914, 0.4822, 0.4465]
  normalize_std = [0.2023, 0.1994, 0.2010]


  #indices = torch.randperm(len(testset))[:300]

  trainset=datasets.CIFAR10(
          args.data_dir,
          train=True,
          download=True,
          transform=transforms.Compose([
              transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize(normalize_mean, normalize_std)]))
  
  testset=datasets.CIFAR10(
          args.data_dir,
          train=False,
          download=True,
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(normalize_mean, normalize_std)]))



  indices = torch.randperm(len(trainset))#[1:1000]
  validset =torch.utils.data.Subset(trainset, indices[int(0.9*len(indices)):(int(1*len(indices))-1)])
  trainset =torch.utils.data.Subset(trainset, indices[:int(0.9*len(indices))])

  train_loader = torch.utils.data.DataLoader(trainset,
      batch_size=128, shuffle=True, drop_last=False, **kwargs)


  valid_loader = torch.utils.data.DataLoader(validset ,
      batch_size=128, shuffle=True, drop_last=False, **kwargs)

  test_loader = torch.utils.data.DataLoader(testset,
      batch_size=1000, shuffle=False, drop_last=False, **kwargs)


  print('Successfully get data loader.')

  run(args, device, train_loader,valid_loader,test_loader)


if __name__ == '__main__':
  main(parse_flags())
