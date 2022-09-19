import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torchvision import models, datasets, transforms
from TaskModels_Pretrained import *


def get_CIFAR10(root="./cifar_test/"):
    input_size = 32
    num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, test_dataset


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=10)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x


def train(model, train_loader, optimizer, epoch,criterion):
    model.train()

    total_loss = []

    for data, target in tqdm(train_loader):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        prediction = model(data)
        loss = criterion(prediction, target)

        #loss = F.nll_loss(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")


def test(model, test_loader,criterion):
    model.eval()

    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            prediction = model(data)
            loss += criterion(prediction, target)

            #loss += F.nll_loss(prediction, target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    return loss, percentage_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="learning rate (default: 0.05)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

    #--------------------------------

    parser.add_argument('--p', type=float, default=0.5,
					help='probability of dropout')


    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42).')

    parser.add_argument('--Hidden_dim', type=int, default=50,
                        help='hidden dim of NN')


    parser.add_argument('--Data', type=str, default='MNIST',
                        help='Which data to use')

    parser.add_argument('--Method', type=str, default='Original',
                        help='dropout method')

    parser.add_argument('--Epochs', type=int, default=200,
                        help='Number of epochs')

    parser.add_argument('--RewardType', type=int, default=0,
                        help='0:only training set, 1:validation set , 2: validation set +augmentation')

    parser.add_argument('--DataRatio', type=float, default=1.0,
                        help='ratio of data used for the training (0-1), for small data regime experiments')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='how sharp the reward for GFN is')
    parser.add_argument('--folder', type=str, default='Results',
                        help='Folder to store the results of the experiments')


    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size, num_classes, train_dataset, test_dataset = get_CIFAR10()

    kwargs = {"num_workers": 2, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True,**kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=5000, shuffle=False, **kwargs
    )

    #model = Model()
    model  = ResNet(num_layers=18,img_channels=3)
    model = model.to(device)

    milestones = [25, 40]
    criterion =  nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch,criterion)
        test(model, test_loader,criterion)
        
        scheduler.step()

    torch.save(model.state_dict(), "cifar_model.pt")


if __name__ == "__main__":
    main()