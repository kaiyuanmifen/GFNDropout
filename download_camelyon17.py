from wilds import get_dataset
from torchvision import transforms

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = get_dataset(dataset="camelyon17", download=True, root_dir ='data')
print('Getting sub dataset')
trainset = dataset.get_subset("train", transform=transform)
testset = dataset.get_subset("test", transform=transform)
validset = dataset.get_subset("val", transform=transform)
print('Dataset over')
