import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
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
parser.add_argument('--valid', type=bool, default=True)
args = parser.parse_args()

logger = Logger('mnist-{}'.format(args.mg_type), tensorboard_dir='tensorboard')
logger.add_params(vars(args))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0), (1))])
train = datasets.MNIST(root='data/', train=True, download=True, transform=transform)
test = datasets.MNIST(root='data/', train=False, transform=transform)
indices = torch.randperm(len(train))[:(2700 if args.valid else 3000)]
trainset = torch.utils.data.Subset(train, indices)
indices = torch.randperm(len(test))[:300]
testset = torch.utils.data.Subset(test, indices)
if args.valid:
    indices = torch.randperm(len(train))[:300]
    validset = torch.utils.data.Subset(train, indices)
else:
    validset = testset  # never used

trainloader = torch.utils.data.DataLoader(trainset, batch_size=30, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=30, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

model = BaysianMLPClassifierWithMaskGenerator(
    in_dim=784,
    out_dim=10,
    hidden=[128, 128,32,128,64,128,256,128],
    activation=nn.LeakyReLU,
    dropout_rate=0.1,
    mg_type=args.mg_type,
    lr=1e-3,
    z_lr=1e-1,
    mg_lr=1e-3,
    mg_hidden=None,
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

Data_set_size=90
for epoch in range(50):
    for data in tqdm(trainloader):
        x, y = data
        if args.valid:
            x_valid, y_valid = next(iter(validloader))
        else:
            x_valid, y_valid = None, None
        metric = model.step(x.reshape((-1, 784)), y,Data_set_size)
        for k in metric.keys():
            logger.add_metric('train_{}'.format(k), metric[k])
    for data in tqdm(testloader):
        x, y = data
        metric,masks = model.test(x.reshape((-1, 784)), y)

        for k in metric.keys():
            logger.add_metric('test_{}'.format(k), metric[k])



        if epoch in [0,5,10,20,40,100,150,200]:
            ###occasionally save the mask distrbution
            MeanPercentage=(masks.sum(2)/masks.shape[2]).mean().item()

            VecPlot=(masks.sum(2)/masks.shape[2])[2,:].tolist()

            fig=plt.figure(figsize=(8, 7))

            plt.plot(range(1, len(VecPlot)+1), VecPlot, '.', alpha=0.6);
            plt.ylim([-0.1, 1.2]);
            #plt.legend(loc=1);
            plt.xlabel('unit');
            plt.ylabel('% dropout over 5 repeats');
            plt.title( str(epoch)+"epochs on average "+str(100*MeanPercentage)+'% of dropout of \n a single data point over 5 samples of masks')
            plt.savefig('MasksSamples_'+str(epoch)+'_epoch.png')
            plt.clf()


            ####histogram
            fig=plt.figure(figsize=(8, 7))

            hist, bins = np.histogram(VecPlot, bins=50)

            #width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            
            plt.bar(center, hist, align='center')
 
            plt.xlabel('dropout rate');
            plt.ylabel('freq');
            plt.title( str(epoch)+"epochs on average "+str(100*MeanPercentage)+'% of dropout of \n a single data point over 5 samples of masks')
            plt.savefig('HistogramMasksSamples_'+str(epoch)+'_epoch.png')
            plt.clf()



    logger.commit(epoch=epoch, step=epoch * len(trainloader))
