import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from model import MLPClassifierWithMaskGenerator
from logger import Logger

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

model = MLPClassifierWithMaskGenerator(
    in_dim=784,
    out_dim=10,
    hidden=[32, 32],
    activation=nn.LeakyReLU,
    dropout_rate=0.5,
    mg_type=args.mg_type,
    lr=1e-3,
    z_lr=1e-1,
    mg_lr=1e-3,
    mg_hidden=None,
    mg_activation=nn.LeakyReLU,
    beta=0.1,
    device='cpu',
)


for epoch in range(300):
    for data in tqdm(trainloader):
        x, y = data
        if args.valid:
            x_valid, y_valid = next(iter(validloader))
        else:
            x_valid, y_valid = None, None
        metric = model.step(x.reshape((-1, 784)), y, x_valid.reshape((-1, 784)), y_valid)
        for k in metric.keys():
            logger.add_metric('train_{}'.format(k), metric[k])
    for data in tqdm(testloader):
        x, y = data
        metric = model.test(x.reshape((-1, 784)), y)
        for k in metric.keys():
            logger.add_metric('test_{}'.format(k), metric[k])
    logger.commit(epoch=epoch, step=epoch * len(trainloader))
