import torch
from torchvision import transforms, datasets
from config import opt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# dir = '/work/06008/xf993/maverick2/contextual_dropout'
dir = '../'


def flip_target(targets, flip_prob, num_classes):
    np.random.seed(1)
    noise_mask = torch.from_numpy(np.random.uniform(size=targets.shape) < flip_prob)
    targets[noise_mask] = torch.from_numpy(np.random.choice(num_classes, size=targets[noise_mask].shape)).type_as(
        targets)
    return targets


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (
            self.x[index],
            self.y[index],
        )


def iecu(augment=False, batch_size=128):
    path = "/home/mila/b/bonaventure.dossou/iecu_data.csv"
    iecu_dataset = pd.read_csv(path, delimiter=",")
    targets = iecu_dataset["Death"].values
    iecu_dataset.drop(columns=['Death'], inplace=True)
    
    if opt.model == "MLP_GFN":
        features = iecu_dataset.values[:, :1369]
        features = features.reshape(-1, 37, 37)
    else:
        features = iecu_dataset.values

    train_data, eval_test_data, train_labels, eval_test_labels = train_test_split(features, targets, test_size=0.3,
                                                                                  random_state=1234)  # 70:30
    eval_data, test_data, eval_data_labels, test_data_labels = train_test_split(eval_test_data, eval_test_labels, test_size=0.3,
                                                                                random_state=1234) #  70:30

    iecu_train_dataset = Dataset(torch.tensor(train_data).float(), torch.tensor(train_labels))
    iecu_valid_dataset = Dataset(torch.tensor(eval_data).float(), torch.tensor(eval_data_labels))
    iecu_test_dataset = Dataset(torch.tensor(test_data).float(), torch.tensor(test_data_labels))

    train_loader = torch.utils.data.DataLoader(iecu_train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(iecu_valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(iecu_test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 2

    # print(torch.tensor(train_data).float().size())
    return train_loader, validation_loader, test_loader, num_classes


def mnist(batch_size=100, pm=False):
    transf = [transforms.ToTensor()]
    num_classes = 10
    if pm:
        transf.append(transforms.Lambda(lambda x: x.view(-1, 784)))
    transform_data = transforms.Compose(transf)

    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}

    train_dataset = datasets.MNIST(dir + '/data', train=True, download=True,
                                   transform=transform_data)
    if opt.labelnoise_train > 0.0:
        train_dataset.targets = flip_target(train_dataset.targets, opt.labelnoise_train, num_classes)

    ##subset of training for speed when needed
    indices = list(range(len(train_dataset)))  # [0::5]
    train_dataset = torch.utils.data.Subset(train_dataset, indices)

    # s subset of trainingset is used as validation set for purposes such as early stopping
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [int(0.7 * len(train_dataset)),
                                                                                      int(0.3 * len(train_dataset))])

    train_loader = torch.utils.data.DataLoader(train_dataset
                                               ,
                                               batch_size=batch_size, shuffle=True, **kwargs)

    validation_loader = torch.utils.data.DataLoader(validation_dataset
                                                    ,
                                                    batch_size=batch_size, shuffle=False, **kwargs)

    # test set
    test_dataset = datasets.MNIST(dir + '/data', train=False, transform=transform_data)

    indices = list(range(len(test_dataset)))  # [0::5]
    test_dataset = torch.utils.data.Subset(test_dataset, indices)

    if opt.labelnoise_val > 0.0:
        test_dataset.targets = flip_target(val_dataset.targets, opt.labelnoise_val, num_classes)

    test_loader = torch.utils.data.DataLoader(test_dataset
                                              ,
                                              batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, validation_loader, test_loader, num_classes


def cifar10(augment=True, batch_size=128):
    # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    logging = 'Using'
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        logging += ' augmented'
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    print(logging + ' CIFAR 10.')
    kwargs = {'num_workers': 1, 'pin_memory': torch.cuda.is_available()}

    trainset = datasets.CIFAR10(dir + '/data', train=True, download=True,
                                transform=transform_train)

    indices = list(range(len(trainset)))  # [0::5]
    trainset = torch.utils.data.Subset(trainset, indices)

    trainset, validation_dataset = torch.utils.data.random_split(trainset,
                                                                 [int(0.7 * len(trainset)), int(0.3 * len(trainset))])

    # subset of training for speed when needed
    indices = list(range(len(trainset)))  # [0::50]
    trainset = torch.utils.data.Subset(trainset, indices)

    testset = datasets.CIFAR10(dir + '/data', train=False, transform=transform_test)
    indices = list(range(len(testset)))  # [0::10]
    testset = torch.utils.data.Subset(testset, indices)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size, shuffle=False, **kwargs)

    val_loader = torch.utils.data.DataLoader(validation_dataset,
                                             batch_size=batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size, shuffle=False, **kwargs)

    num_classes = 10

    return train_loader, val_loader, test_loader, num_classes


def cifar100(augment=True, batch_size=128):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                     std=[x / 255.0 for x in [68.2, 65.4, 70.4]])

    logging = 'Using'
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        logging += ' augmented'
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    print(logging + ' CIFAR 100.')
    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(dir + '/data', train=True, download=True,
                          transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(dir + '/data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=True, **kwargs)
    num_classes = 100

    return train_loader, val_loader, num_classes