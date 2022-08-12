import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch import autograd
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from model import BaysianMLPClassifierWithMaskGenerator
from logger import Logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')

parser = argparse.ArgumentParser()
parser.add_argument('--mg_type', type=str, default='gfn')
parser.add_argument('--task', type=str)
args = parser.parse_args()

logger = Logger('mnist-{}'.format(args.mg_type), tensorboard_dir='tensorboard')
logger.add_params(vars(args))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0), (1))])
train = datasets.MNIST(root='data/', train=True, download=True, transform=transform)
test = datasets.MNIST(root='data/', train=False, transform=transform)
indices = torch.randperm(len(train))#[:10000]#[:(9700 if args.valid else 10000)]

train_indices=indices[:int(0.7*len(indices))]
valid_indices=indices[int(0.7*len(indices)):]
trainset = torch.utils.data.Subset(train, train_indices)
validset = torch.utils.data.Subset(train, valid_indices)

test_indices = torch.randperm(len(test))#[:3000]
testset = torch.utils.data.Subset(test, test_indices)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=256, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

model = BaysianMLPClassifierWithMaskGenerator(
    in_dim=784,
    out_dim=10,
    hidden=[128,128],
    activation=nn.LeakyReLU,
    dropout_rate=0.1,
    mg_type=args.mg_type,
    lr=1e-3,
    z_lr=1e-1,
    mg_lr=1e-3,
    mg_hidden=[128,128],
    mg_activation=nn.LeakyReLU,
    beta=0.1,
    device='cpu',
)


print("****mmodel")
print(model.model)
for param in model.model.parameters():
    param.requires_grad = True

model.model.out_layer.requires_grad_(True)


print("trainable parameters")
for name, param in model.model.named_parameters():
    if param.requires_grad:
        print (name)





def train_base(N_epochs):
        #train the base BNN or simple MLP model


        Data_set_size=len(train_indices)
        metric_plot = {}
        train_BNN=True
        mask_off=True
        best_valid_loss=1e9
            
        for epoch in range(1,N_epochs):


            for data in tqdm(trainloader):

                x, y = data
                # if args.valid:
                #     x_valid, y_valid = next(iter(validloader))
                # else:
                #     x_valid, y_valid = None, None

                if train_BNN:
                    #run BNN step first to train the BNN model

                    metric = model.BNNstep(x.reshape((-1, 784)), y,Data_set_size,mask_off)
                
                else:
                    #run GFN step to train the BNN model, topdown and bottom up mask together

                    metric = model.GFNstep(x.reshape((-1, 784)), y,Data_set_size,mask_off)

           
                
                for k in metric.keys():
                    logger.add_metric('base_train_{}'.format(k), metric[k])

            #validation
            model.eval()
            val_loss=0
            val_acc=0
            for data in tqdm(validloader):
                x, y = data
                metric,masks,LogZ_mu_individual,Log_PF_mu_individual,R_mu_individual,Log_PF_BNN_individual = model.test(x.reshape((-1, 784)), y,len(valid_indices),0)
                val_loss+=metric['CELoss']
                for k in metric.keys():
                    logger.add_metric('base_valid_{}'.format(k), metric[k])


            if metric['CELoss']<best_valid_loss:
                best_valid_loss=metric['CELoss']
                torch.save(model.state_dict(), "checkpoints/base_Model.pt")


            logger.commit(epoch=epoch, step=epoch * len(trainloader))



def train_masked(N_epochs):

        
        ###start from pretrained 
        print('loading saved model')
        model.load_state_dict(torch.load("checkpoints/base_Model.pt"))
        model.train()
        ###train the mask generator

        Data_set_size=len(train_indices)
        metric_plot = {}
        train_BNN=True
        mask_off=True
        best_valid_loss=1e9
            
        for epoch in range(1,N_epochs):
            model.train()
            # if epoch%40==0:
            #     #switch between training every 10 epochs
            #     train_BNN=not train_BNN
            #     mask_off=False
            if epoch<0:
                #switch between training every 10 epochs
                train_BNN=True
                mask_off=True
            else:

                train_BNN=False
                mask_off=False

            for data in tqdm(trainloader):

                x, y = data
                # if args.valid:
                #     x_valid, y_valid = next(iter(validloader))
                # else:
                #     x_valid, y_valid = None, None

                if train_BNN:
                    #run BNN step first to train the BNN model

                    metric = model.BNNstep(x.reshape((-1, 784)), y,Data_set_size,mask_off)
                
                else:
                    #run GFN step to train the BNN model, topdown and bottom up mask together

                    metric = model.GFNstep(x.reshape((-1, 784)), y,Data_set_size,mask_off)

           
                
                for k in metric.keys():
                    logger.add_metric('masked_train_{}'.format(k), metric[k])

            #validation
            model.eval()
            val_loss=0
            val_acc=0
            for data in tqdm(validloader):
                x, y = data
                metric,masks,LogZ_mu_individual,Log_PF_mu_individual,R_mu_individual,Log_PF_BNN_individual = model.test(x.reshape((-1, 784)), y,len(valid_indices),0)
                val_loss+=metric['CELoss']
                for k in metric.keys():
                    logger.add_metric('masked_valid_{}'.format(k), metric[k])


            if metric['CELoss']<best_valid_loss:
                best_valid_loss=metric['CELoss']
                torch.save(model.state_dict(), "checkpoints/masked_Model.pt")


            logger.commit(epoch=epoch, step=epoch * len(trainloader))



def test(task,repeats=1):


        metric_plot = {}
        ####load pre-saved model 
        if args.task=='testbase':
          print('loading saved base model')
          model.load_state_dict(torch.load("checkpoints/base_Model.pt"))
          model.eval()

        elif args.task=='testmasked':
          print('loading saved mased model')
          model.load_state_dict(torch.load("checkpoints/masked_Model.pt"))
          model.eval()


        #validation
        data= next(iter(testloader))
        x, y = data

        metric,masks,LogZ_mu_individual,Log_PF_mu_individual,LogR_mu_individual,Log_PF_BNN_individual = model.test(x=x.reshape((-1, 784)), y=y,Data_set_size=len(testset),
                                                                                            masks_off=False,epsilon=0.0,SampleRepeats=repeats)
        
 
        import pandas as pd

        testResults=pd.DataFrame({"LogPF":Log_PF_mu_individual.tolist(),"LogRmu":LogR_mu_individual.tolist(),"LogZmu":LogZ_mu_individual.tolist(),"LogPF_BNN":Log_PF_BNN_individual.tolist()})
        testResults.to_csv("images/"+args.task+"_testResults.csv")
        print(metric)

         
        if args.task=='testmasked':
          ####plotting
          ###P_F_mu vs R_mu


          fig=plt.figure(figsize=(8, 7))

          x=Log_PF_mu_individual
          y=LogR_mu_individual
          

          coef = np.polyfit(x,y,1)
          poly1d_fn = np.poly1d(coef) 
          plt.plot(x, y,  'yo', x, poly1d_fn(x), '--k', alpha=0.6);
          #plt.plot(x, y,  '.', alpha=0.6);
          from scipy.stats.stats import pearsonr

          Cor=pearsonr(x.numpy(),y.numpy())[0]


          #plt.ylim([-0.1, 1.2]);
          #plt.legend(loc=1);
          plt.xlabel('PF_mu');
          plt.ylabel('exp R_mu');
          plt.title( "PF_mu vs Log R_mu , Pearson Cor: "+str(Cor))
          plt.savefig('images/PF_Log_R_mu_'+'_epoch.png')
          plt.clf()

          ###P_F_BNN vs R_mu


          fig=plt.figure(figsize=(8, 7))

          x=Log_PF_BNN_individual
          y=LogR_mu_individual
          

          coef = np.polyfit(x,y,1)
          poly1d_fn = np.poly1d(coef) 
          plt.plot(x, y,  'yo', x, poly1d_fn(x), '--k', alpha=0.6);
          #plt.plot(x, y,  '.', alpha=0.6);
          from scipy.stats.stats import pearsonr

          Cor=pearsonr(x.numpy(),y.numpy())[0]


          #plt.ylim([-0.1, 1.2]);
          #plt.legend(loc=1);
          plt.xlabel('log PF_BNN');
          plt.ylabel('log R_mu');
          plt.title( "log PF_BNN vs Log R_mu , Pearson Cor: "+str(Cor))
          plt.savefig('images/log PF BNN _Log_R_mu_'+'_epoch.png')
          plt.clf()


          ###occasionally save the mask distrbution
          MeanPercentage=((masks==0).sum(2)/masks.shape[2]).mean().item()

          VecPlot=((masks==0).sum(2)/masks.shape[2])[2,:].tolist()

          fig=plt.figure(figsize=(8, 7))

          plt.plot(range(1, len(VecPlot)+1), VecPlot, '.', alpha=0.6);
          plt.ylim([-0.1, 1.2]);
          #plt.legend(loc=1);
          plt.xlabel('unit');
          plt.ylabel('% dropout over 5 repeats');
          plt.title( "DropoutRate_"+str(100*MeanPercentage)+'% of dropout of \n a single data point over 10 samples of masks')
          plt.savefig('images/MasksSamples_Dropout'+'_epoch.png')
          plt.clf()


          ####histogram
          fig=plt.figure(figsize=(8, 7))

          hist, bins = np.histogram(VecPlot, bins=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

          #width = 0.7 * (bins[1] - bins[0])
          center = (bins[:-1] + bins[1:]) / 2
          
          plt.bar(center, hist, align='center')
          plt.xlim(0, 1)
          plt.xlabel('dropout rate');
          plt.ylabel('freq');
          plt.title( "Dropout on average "+str(100*MeanPercentage)+'% of dropout of \n a single data point over 10 samples of masks')
          plt.savefig('images/HistogramMasksSamples_epoch.png')
          plt.clf()






if args.task=="trainbase":
    train_base(200)

elif args.task=="trainmasked":
    train_masked(200)

elif args.task=="testbase":
    test("base")

elif args.task=="testmasked":
    test("masked")


