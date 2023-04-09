import torch
from torchvision import transforms, datasets
from config import opt
import numpy as np
from PIL import Image
import os
#dir = '/work/06008/xf993/maverick2/contextual_dropout'
#dir = '/home/mila/d/dianbo.liu/scratch/GFlownets/data/'
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


	##subset of training for speed when needed
	indices = list(range(len(train_dataset)))#[0::5]
	train_dataset = torch.utils.data.Subset(train_dataset, indices)

	# s subset of trainingset is used as validation set for purposes such as early stopping
	train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [int(0.7*len(train_dataset)), int(0.3*len(train_dataset))])

	train_loader = torch.utils.data.DataLoader(train_dataset
		,
		batch_size=batch_size, shuffle=True, **kwargs)

	validation_loader = torch.utils.data.DataLoader(validation_dataset
		,
		batch_size=batch_size, shuffle=False,drop_last=True, **kwargs)



	#test set
	test_dataset = datasets.MNIST(dir+'/data', train=False, transform=transform_data) 
	
	indices = list(range(len(test_dataset)))#[0::5]
	test_dataset = torch.utils.data.Subset(test_dataset, indices)
	
	if opt.labelnoise_val > 0.0:
		test_dataset.targets = flip_target(val_dataset.targets, opt.labelnoise_val, num_classes)

	test_loader = torch.utils.data.DataLoader(test_dataset
		,
		batch_size=batch_size, shuffle=False,drop_last=True, **kwargs)
	

	return train_loader,validation_loader, test_loader, num_classes


def cifar10( batch_size=128,subset_size=None):
	#normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
	 #                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

	augment=True
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



	trainset=datasets.CIFAR10(dir+'/data', train=True, download=True,
						 transform=transform_train)
	#subset of training for speed when needed
	indices = list(range(len(trainset)))#[0::5]
	if subset_size!=None:
		Skip=int(len(indices)/(subset_size*(1/0.7)))
		indices = list(range(len(trainset)))[0::Skip]
	trainset = torch.utils.data.Subset(trainset, indices)

	trainset, validation_dataset = torch.utils.data.random_split(trainset, [int(0.7*len(trainset)), len(trainset)-int(0.7*len(trainset))])

	print("subset_size",subset_size)
	print("trainset",len(trainset))
	print("validation_dataset",len(validation_dataset))


	# indices = list(range(len(trainset)))#[0::50]
	# trainset = torch.utils.data.Subset(trainset, indices)

	testset=datasets.CIFAR10(dir+'/data', train=False, transform=transform_test,download=True,)
	indices = list(range(len(testset)))#[0::100]
	testset = torch.utils.data.Subset(testset, indices)


	train_loader = torch.utils.data.DataLoader(trainset,
		batch_size=batch_size, shuffle=False, **kwargs)

	val_loader = torch.utils.data.DataLoader(validation_dataset,
		batch_size=batch_size, shuffle=False,drop_last=True, **kwargs)


	test_loader = torch.utils.data.DataLoader(testset,
		batch_size=batch_size, shuffle=False,drop_last=True, **kwargs)
	
	num_classes = 10

	return train_loader, val_loader,test_loader, num_classes


def cifar100( batch_size=128,subset_size=None):
	normalize = transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
									 std=[x / 255.0 for x in [68.2, 65.4, 70.4]])
	augment=True
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


	trainset=datasets.CIFAR100(dir+'/data', download=True,train=True,
						 transform=transform_train)


	#subset of training for speed when needed
	indices = list(range(len(trainset)))#[0::50]
	if subset_size!=None:
		Skip=int(len(indices)/(subset_size*(1/0.7)))
		indices = list(range(len(trainset)))[0::Skip]
	trainset = torch.utils.data.Subset(trainset, indices)
	
	trainset, validation_dataset = torch.utils.data.random_split(trainset, [int(0.7*len(trainset)), len(trainset)-int(0.7*len(trainset))])
	
	print("trainset size")
	print(len(trainset))

	print("validation size")
	print(len(validation_dataset))

	train_loader = torch.utils.data.DataLoader(
		trainset,
		batch_size=batch_size, shuffle=True, **kwargs)


	val_loader = torch.utils.data.DataLoader(
		validation_dataset,
		batch_size=batch_size, shuffle=True,drop_last=True, **kwargs)
	
	testset=datasets.CIFAR100(dir+'/data', train=False, transform=transform_test,download=True,)
	indices = list(range(len(testset)))#[0::100]
	testset = torch.utils.data.Subset(testset, indices)

	test_loader = torch.utils.data.DataLoader(testset,
		batch_size=batch_size, shuffle=False,drop_last=True, **kwargs)

	num_classes = 100

	return train_loader, val_loader,test_loader, num_classes





def LFWPeople(augment=False, batch_size=128):
	normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

	logging = 'Using'
	if augment:
		transform_train = transforms.Compose([
			#transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.Resize((32,32)),
			transforms.ToTensor(),
			normalize,
			])
		logging += ' augmented'
	else:
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize((32, 32)),
			normalize,
			])
	transform_test = transforms.Compose([

		transforms.ToTensor(),
		transforms.Resize((32, 32)),
		normalize
		])

	print(logging + 'LFWPeople.')
	kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}


	trainset=datasets.LFWPeople(dir+'/data', split="train", download=True,
						 transform=transform_train)


	#subset of training for speed when needed
	indices = list(range(len(trainset)))#[0::50]
	trainset = torch.utils.data.Subset(trainset, indices)
	
	trainset, validation_dataset = torch.utils.data.random_split(trainset, [int(0.7*len(trainset)), len(trainset)-int(0.7*len(trainset))])
	



	train_loader = torch.utils.data.DataLoader(
		trainset,
		batch_size=batch_size, shuffle=True, **kwargs)


	val_loader = torch.utils.data.DataLoader(
		validation_dataset,
		batch_size=batch_size, shuffle=True, **kwargs)
	
	testset=datasets.LFWPeople(dir+'/data', split="test", transform=transform_test,download=True)
	indices = list(range(len(testset)))#[0::100]
	testset = torch.utils.data.Subset(testset, indices)

	test_loader = torch.utils.data.DataLoader(testset,
		batch_size=batch_size, shuffle=False, **kwargs)

	num_classes = 5749


	return train_loader, val_loader,test_loader, num_classes




def FER2013(batch_size=100, pm=False):
	transf = [transforms.ToTensor()]
	num_classes = 10
	if pm:
		transf.append(transforms.Lambda(lambda x: x.view(-1, 784)))
	transform_data = transforms.Compose(transf)

	kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}

	train_dataset = datasets.FER2013(dir+'/data', train=True, download=True,
					   transform=transform_data)
	if opt.labelnoise_train > 0.0:
		train_dataset.targets = flip_target(train_dataset.targets, opt.labelnoise_train, num_classes)


	##subset of training for speed when needed
	indices = list(range(len(train_dataset)))#[0::5]
	train_dataset = torch.utils.data.Subset(train_dataset, indices)

	# s subset of trainingset is used as validation set for purposes such as early stopping
	train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [int(0.7*len(train_dataset)), int(0.3*len(train_dataset))])

	train_loader = torch.utils.data.DataLoader(train_dataset
		,
		batch_size=batch_size, shuffle=True, **kwargs)

	validation_loader = torch.utils.data.DataLoader(validation_dataset
		,
		batch_size=batch_size, shuffle=False, **kwargs)



	#test set
	test_dataset = datasets.FER2013(dir+'/data', split="train", transform=transform_data) 
	
	indices = list(range(len(test_dataset)))#[0::5]
	test_dataset = torch.utils.data.Subset(test_dataset, indices)
	
	if opt.labelnoise_val > 0.0:
		test_dataset.targets = flip_target(val_dataset.targets, opt.labelnoise_val, num_classes)

	test_loader = torch.utils.data.DataLoader(test_dataset
		,
		batch_size=batch_size, shuffle=False, **kwargs)
	

	return train_loader,validation_loader, test_loader, num_classes



###CIFAR10-C


def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]


corruptions = load_txt(dir+'/data/CIFAR-10-C/'+'corruptions.txt')

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root=dir+'/data/CIFAR-10-C/', name="defocus_blur",severity=1,
                 transform=None, target_transform=None):
        assert name in corruptions
        super(CIFAR10C, self).__init__(
            root=root,transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.data=self.data[((severity-1)*10000):(severity*10000),:,:,:]

        self.targets = np.load(target_path)

        self.targets =self.targets [((severity-1)*10000):(severity*10000)]
   
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        
        targets=torch.tensor(targets).long()
        return img, targets
    
    def __len__(self):
        return len(self.data)


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)



def cifar10c( batch_size=128,name="defocus_blur",severity=1):
	normalize = transforms.Normalize(mean= [0.49139968, 0.48215841, 0.44653091],
	                                std=[0.24703223, 0.24348513, 0.26158784])


	logging = 'Using'

	transform= transforms.Compose([
		transforms.ToTensor(),
		normalize,
		])

	

	print(logging + ' CIFAR 10C')
	kwargs = {'num_workers': 1, 'pin_memory': torch.cuda.is_available()}


	dataset=CIFAR10C(root=dir+'/data/CIFAR-10-C/',name=name,severity=severity,transform=transform)

	# indices = list(range(len(dataset)))[0::100]
	# dataset = torch.utils.data.Subset(dataset, indices)

	trainset, validation_dataset ,testset=dataset,dataset,dataset #all the same , just placeholders to main the same format as others


	train_loader = torch.utils.data.DataLoader(trainset,
		batch_size=batch_size, shuffle=False,drop_last=True, **kwargs)

	val_loader = torch.utils.data.DataLoader(validation_dataset,
		batch_size=batch_size, shuffle=False,drop_last=True, **kwargs)


	test_loader = torch.utils.data.DataLoader(testset,
		batch_size=batch_size, shuffle=False,drop_last=True, **kwargs)
	
	num_classes = 10

	return train_loader, val_loader,test_loader, num_classes




####cifar100-c

class CIFAR100C(datasets.VisionDataset):
    def __init__(self, root=dir+'/data/CIFAR-100-C/', name="defocus_blur",severity=1,
                 transform=None, target_transform=None):
        assert name in corruptions
        super(CIFAR100C, self).__init__(
            root=root,transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.data=self.data[((severity-1)*10000):(severity*10000),:,:,:]

        self.targets = np.load(target_path)

        self.targets =self.targets [((severity-1)*10000):(severity*10000)]
   
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        
        targets=torch.tensor(targets).long()
        return img, targets
    
    def __len__(self):
        return len(self.data)




def cifar100c( batch_size=128,name="defocus_blur",severity=1):
	normalize = transforms.Normalize(mean= [0.49139968, 0.48215841, 0.44653091],
	                                std=[0.24703223, 0.24348513, 0.26158784])


	logging = 'Using'

	transform= transforms.Compose([
		transforms.ToTensor(),
		normalize,
		])

	

	print(logging + ' CIFAR 100C')
	kwargs = {'num_workers': 1, 'pin_memory': torch.cuda.is_available()}


	dataset=CIFAR100C(root=dir+'/data/CIFAR-100-C/',name=name,severity=severity,transform=transform)

	# indices = list(range(len(dataset)))[0::500]
	# dataset = torch.utils.data.Subset(dataset, indices)
	
	trainset, validation_dataset ,testset=dataset,dataset,dataset #all the same , just placeholders to main the same format as others


	train_loader = torch.utils.data.DataLoader(trainset,
		batch_size=batch_size, shuffle=False, **kwargs)

	val_loader = torch.utils.data.DataLoader(validation_dataset,
		batch_size=batch_size, shuffle=False,drop_last=True, **kwargs)


	test_loader = torch.utils.data.DataLoader(testset,
		batch_size=batch_size, shuffle=False,drop_last=True, **kwargs)
	
	num_classes = 100

	return train_loader, val_loader,test_loader, num_classes



