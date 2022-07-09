import torch
from torchvision import transforms, datasets
from config import opt
import numpy as np
#dir = '/work/06008/xf993/maverick2/contextual_dropout'
dir = '../../../../'

def flip_target(targets, flip_prob, num_classes):
    np.random.seed(1)
    noise_mask = torch.from_numpy(np.random.uniform(size=targets.shape) < flip_prob)
    targets[noise_mask] = torch.from_numpy(np.random.choice(num_classes, size=targets[noise_mask].shape)).type_as(targets)
    return targets

def mnist(batch_size=100, pm=False):
    transf = [transforms.ToTensor()]
    num_classes = 10
    if pm:
        transf.append(transforms.Lambda(lambda x: x.view(-1, 784)))
    transform_data = transforms.Compose(transf)

    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}

    train_dataset = datasets.MNIST(dir+'/data', train=True, download=True,
                       transform=transform_data)
    if opt.labelnoise_train > 0.0:
        train_dataset.targets = flip_target(train_dataset.targets, opt.labelnoise_train, num_classes)

    train_loader = torch.utils.data.DataLoader(train_dataset
        ,
        batch_size=batch_size, shuffle=True, **kwargs)

    val_dataset = datasets.MNIST(dir+'/data', train=False, transform=transform_data) 

    if opt.labelnoise_val > 0.0:
        val_dataset.targets = flip_target(val_dataset.targets, opt.labelnoise_val, num_classes)

    val_loader = torch.utils.data.DataLoader(val_dataset
        ,
        batch_size=batch_size, shuffle=False, **kwargs)
    

    return train_loader, val_loader, num_classes


def cifar10(augment=True, batch_size=128):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

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
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(dir+'/data', train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(dir+'/data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=False, **kwargs)
    num_classes = 10

    return train_loader, val_loader, num_classes


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
        datasets.CIFAR100(dir+'/data', train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(dir+'/data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=True, **kwargs)
    num_classes = 100

    return train_loader, val_loader, num_classes