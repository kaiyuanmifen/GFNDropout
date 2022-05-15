from wilds import get_dataset
import torch
import numpy as np
import torch.optim as optim 
from GFN_SampleMask import GFN_SamplingMask

from GFNFunctions import *
from Dropout_DIY import *
from TaskModels import *


import matplotlib.pyplot as plt
import matplotlib

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import random

import numpy as np
import pandas as pd
import time
import h5py
from scipy.ndimage.interpolation import rotate

import argparse

matplotlib.use('AGG')





parser = argparse.ArgumentParser()

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

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)


Task_name=args.Method+"_"+args.Data+"_"+str(args.Hidden_dim)+"_"+str(args.p)+"_"+str(args.beta)+"_"+str(args.RewardType)+"_"+str(args.DataRatio)+"_"+str(args.seed)

print("task:",Task_name)







device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####part 1 load MNIST/CIFAR data
batch_size=128
if args.Data=="MNIST":

	transform = transforms.Compose([transforms.ToTensor(), \
									transforms.Normalize((0), (1))])

	trainset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
	testset = datasets.MNIST(root='data/', train=False, transform=transform)

	indices = torch.randperm(len(trainset))[:int(len(trainset)*args.DataRatio)]
	#indices = torch.randperm(len(trainset))

	validset =torch.utils.data.Subset(trainset, indices[int(0.7*len(indices)):(int(1*len(indices))-1)])
	
	trainset =torch.utils.data.Subset(trainset, indices[:int(0.7*len(indices))])

	#indices = torch.randperm(len(testset))[:300]
	indices = torch.randperm(len(testset))

	testset  =torch.utils.data.Subset(testset, indices)

	print("training set length")
	print(len(trainset))

	print("validation set length")
	print(len(validset))


	print("test set length")
	print(len(testset))

	# Visualize 10 image samples in MNIST dataset
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

	validloader = torch.utils.data.DataLoader(validset, batch_size=len(validset), shuffle=False)
		

	testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
	

	dataiter = iter(trainloader)
	images, labels = dataiter.next()


	# # plot 10 sample images
	_,ax = plt.subplots(1,10)
	ax = ax.flatten()
	iml = images[0].numpy().shape[1]
	[ax[i].imshow(np.transpose(images[i].numpy(),(1,2,0)).reshape(iml,-1),cmap='Greys') for i in range(10)]
	[ax[i].set_axis_off() for i in range(10)]
	plt.savefig('images/MNISTData.png')

	print('label:',labels[:10].numpy())
	print('image data shape:',images[0].numpy().shape)

	# #####augmented version


	#policies = [T.AutoAugmentPolicy.MNIST
	#half augmenters are used for GFN reward, the other half used for OOD testing
	augmenters_train=[transforms.RandomRotation(degrees=(0,90)),
				transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
				transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))]

	augmenters_test=[transforms.RandomRotation(degrees=(90, 180)),
				transforms.RandomResizedCrop(size=(28, 28)),
				transforms.RandomPerspective(distortion_scale=0.6, p=1.0)]
		

	#augmenter = transforms.RandomRotation(degrees=(0, 180))
	for idx,augmenter in enumerate(augmenters_train):
		#rotated_imgs = [rotater(orig_img) for _ in range(4)]
		Augmented_imgs=[augmenter(images[j]).unsqueeze(0) for j in range(10)]
		Augmented_imgs=torch.cat(Augmented_imgs,0)

		iml = Augmented_imgs[0].numpy().shape[1]
		[ax[i].imshow(np.transpose(Augmented_imgs[i].numpy(),(1,2,0)).reshape(iml,-1),cmap='Greys') for i in range(10)]
		[ax[i].set_axis_off() for i in range(10)]
		plt.savefig('images/AugmentedMNIST_'+str(idx)+'.png')


	# imgs = [
	# [augmenter(orig_img) for _ in range(4)]
	# for augmenter in augmenters
	# ]
if args.Data=="CIFAR10":

	transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='data', train=True,
											download=True, transform=transform)


	testset = torchvision.datasets.CIFAR10(root='data', train=False,
										   download=True, transform=transform)
	
	#indices = torch.randperm(len(trainset))[:1000]
	#indices = torch.randperm(len(trainset))
	indices = torch.randperm(len(trainset))[:int(len(trainset)*args.DataRatio)]
	#indices = torch.randperm(len(trainset))

	validset =torch.utils.data.Subset(trainset, indices[int(0.7*len(indices)):(int(1*len(indices)))])
	
	trainset =torch.utils.data.Subset(trainset, indices[:int(0.7*len(indices))])
	#indices = torch.randperm(len(testset))[:300]
	indices = torch.randperm(len(testset))

	testset  =torch.utils.data.Subset(testset, indices)
	
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True)

	validloader = torch.utils.data.DataLoader(validset, batch_size=len(validset), shuffle=False)

	testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
											 shuffle=False, num_workers=2)


	print("training set length")
	print(len(trainset))

	print("validation set length")
	print(len(validset))


	print("test set length")
	print(len(testset))

	classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	def imshow(img):
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('images/DataExamplesCIFAR.png')

	####show some examples
	dataiter = iter(trainloader)
	images, labels = dataiter.next()
	print("images")
	print(images.shape)

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
	
	#half augmenters are used for GFN reward, the other half used for OOD testing
	augmenters_train=[transforms.RandomRotation(degrees=(0,90)),
				transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
				transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))]
		
	augmenters_test=[transforms.RandomRotation(degrees=(90, 180)),
			transforms.RandomResizedCrop(size=(32, 32)),
			transforms.RandomPerspective(distortion_scale=0.6, p=1.0)]
	

	#augmenter = transforms.RandomRotation(degrees=(0, 180))
	for idx,augmenter in enumerate(augmenters_train):
		#rotated_imgs = [rotater(orig_img) for _ in range(4)]
		Augmented_imgs=[augmenter(images[j]).unsqueeze(0) for j in range(30)]
		Augmented_imgs=torch.cat(Augmented_imgs,0)

		img=torchvision.utils.make_grid(Augmented_imgs)
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('images/AugmentedCIFAR_'+str(idx)+'.png')

if args.Data=="SVHN":

	transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.SVHN(root='data', split="train",
											download=True, transform=transform)


	testset = torchvision.datasets.SVHN(root='data', split="test",
										   download=True, transform=transform)
	

	indices = torch.randperm(len(trainset))[:int(len(trainset)*args.DataRatio)]

	validset =torch.utils.data.Subset(trainset, indices[int(0.7*len(indices)):(int(1*len(indices))-1)])
	
	trainset =torch.utils.data.Subset(trainset, indices[:int(0.7*len(indices))])

	indices = torch.randperm(len(testset))

	testset  =torch.utils.data.Subset(testset, indices)
	
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True, num_workers=2)


	validloader = torch.utils.data.DataLoader(validset, batch_size=len(validset), shuffle=False)
		


	testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
											 shuffle=False, num_workers=2)

	print("training set length")
	print(len(trainset))

	print("validation set length")
	print(len(validset))


	print("test set length")
	print(len(testset))
	# classes = ('plane', 'car', 'bird', 'cat',
	# 		   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	def imshow(img):
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('images/DataExamplesSVHN.png')

	####show some examples
	dataiter = iter(trainloader)
	images, labels = dataiter.next()
	print("images")
	print(images.shape)

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
	
	#half augmenters are used for GFN reward, the other half used for OOD testing
	augmenters_train=[transforms.RandomRotation(degrees=(0,90)),
				transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
				transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3))]

	augmenters_test=[transforms.RandomRotation(degrees=(90, 180)),
			transforms.RandomResizedCrop(size=(32, 32)),
			transforms.RandomPerspective(distortion_scale=0.6, p=1.0)]
		

	#augmenter = transforms.RandomRotation(degrees=(0, 180))
	for idx,augmenter in enumerate(augmenters_train):
		#rotated_imgs = [rotater(orig_img) for _ in range(4)]
		Augmented_imgs=[augmenter(images[j]).unsqueeze(0) for j in range(30)]
		Augmented_imgs=torch.cat(Augmented_imgs,0)

		img=torchvision.utils.make_grid(Augmented_imgs)
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('images/AugmentedSVHN_'+str(idx)+'.png')

if args.Data == "camelyon17":

	transform = transforms.Compose(
	[transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	# Load the full dataset, and download it if necessary
	dataset = get_dataset(dataset="camelyon17", download=True, root_dir='data')
	trainset = dataset.get_subset("train", transform=transform)
	testset = dataset.get_subset("test", transform=transform)
	validset = dataset.get_subset("val", transform=transform)
	

	# indices = torch.randperm(len(trainset))[:int(len(trainset)*args.DataRatio)]

	# validset =torch.utils.data.Subset(trainset, indices[int(0.7*len(indices)):(int(1*len(indices))-1)])
	# trainset =torch.utils.data.Subset(trainset, indices[:int(0.7*len(indices))])

	# indices = torch.randperm(len(testset))

	# testset = torch.utils.data.Subset(testset, indices)
	
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True, num_workers=2)


	validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=False)
	testloader = torch.utils.data.DataLoader(testset, batch_size=64,
											 shuffle=False, num_workers=2)

	print("training set length")
	print(len(trainset))

	print("validation set length")
	print(len(validset))


	print("test set length")
	print(len(testset))
	# classes = ('plane', 'car', 'bird', 'cat',
	# 		   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	def imshow(img):
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('images/DataExamplesCamelyon17.png')

	####show some examples
	dataiter = iter(trainloader)
	images, labels, _ = dataiter.next()
	print("images")
	print(images.shape)

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
	
	#half augmenters are used for GFN reward, the other half used for OOD testing
	augmenters_train=[transforms.RandomRotation(degrees=(0,90)),
				transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
				transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3))]

	augmenters_test=[transforms.RandomRotation(degrees=(90, 180)),
			transforms.RandomResizedCrop(size=(96, 96)),
			transforms.RandomPerspective(distortion_scale=0.6, p=1.0)]
		

	#augmenter = transforms.RandomRotation(degrees=(0, 180))
	for idx,augmenter in enumerate(augmenters_train):
		#rotated_imgs = [rotater(orig_img) for _ in range(4)]
		Augmented_imgs=[augmenter(images[j]).unsqueeze(0) for j in range(30)]
		Augmented_imgs=torch.cat(Augmented_imgs,0)

		img=torchvision.utils.make_grid(Augmented_imgs)
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('images/AugmentedCamelyon17_'+str(idx)+'.png')



#part 2 function to run the task

class MLPClassifier:
	def __init__(self, image_size,droprates=0.5, batch_size=128, max_epoch=10, \
				 lr=0.001, momentum=0,model_type="MLP_nodropout",N_units=50):
		# Wrap MLP model
		self.droprates = droprates
		self.batch_size = batch_size
		self.max_epoch = max_epoch
		self.model_type=model_type

		if self.model_type=="MLP_nodropout":
		  #self.model = MLP(Input_size=image_size[0]*image_size[1]*image_size[2], hidden_size=N_units,droprates=0)
		  self.model = MLP_Alldrop(Input_size=image_size[0]*image_size[1]*image_size[2], hidden_size=N_units,droprates=0)

		elif self.model_type=="MLP_dropout":
		  self.model = MLP(Input_size=image_size[0]*image_size[1]*image_size[2], hidden_size=N_units,droprates=droprates)

		elif "MLP_GFN" in self.model_type:
		  self.model = MLP_MaskedDropout(Input_size=image_size[0]*image_size[1]*image_size[2],hidden_size=N_units)

		elif self.model_type=="MLP_SVD":
			self.model = MLP_SVD(Input_size=image_size[0]*image_size[1]*image_size[2],hidden_size=N_units)
		elif self.model_type=="MLP_Standout":
			self.model = MLP_Standout(Input_size=image_size[0]*image_size[1]*image_size[2],hidden_size=N_units,droprates=droprates)
		
		######Faster version of GFN
		elif self.model_type=="MLP_GFFN":#GFFN means faster version of the GFN
			output_dim = 10
			if args.Data == "camelyon17":
				output_dim = 2
			self.model = MLPClassifierWithMaskGenerator(in_dim=image_size[0]*image_size[1]*image_size[2],
														out_dim=output_dim,
														hidden=(N_units,N_units,N_units,N_units),
														activation=nn.LeakyReLU,
														dropout_rate=droprates,
														mg_type='gfn',
														lr=1e-3,
														z_lr=1e-1,
														mg_lr=1e-3,
														mg_hidden=None,
														mg_activation=nn.LeakyReLU,
														beta=1,
														device=device,)

		elif self.model_type == "CNN_GFFN":
			output_dim = 10 if args.Data != "camelyon17" else 2
			self.model = CNNClassifierWithMaskGenerator(in_dim=image_size[0]*image_size[1]*image_size[2],
														out_dim=output_dim,
														hidden=[(3, 32), (32, 32), (32, 64)],
														activation=nn.LeakyReLU,
														dropout_rate=droprates,
														mg_type='gfn',
														lr=1e-3,
														z_lr=1e-1,
														mg_lr=1e-3,
														mg_hidden=None,
														mg_activation=nn.LeakyReLU,
														beta=1,
														device=device,)

		elif self.model_type=="MLP_dropoutAll":
			#dropour on all layers
		  self.model = MLP_Alldrop(Input_size=image_size[0]*image_size[1]*image_size[2], hidden_size=N_units,droprates=droprates)

		elif self.model_type=="MLP_SVDAll":
			self.model = MLP_SVDAll(Input_size=image_size[0]*image_size[1]*image_size[2],hidden_size=N_units)
		
		elif self.model_type=="MLP_StandoutAll":
			self.model = MLP_StandoutAll(Input_size=image_size[0]*image_size[1]*image_size[2],hidden_size=N_units,droprates=droprates)
		



		###CNN

		elif self.model_type=="CNN_nodropout":
		  self.model = CNN(image_size=image_size,hidden_size=N_units,droprates=0)
		elif self.model_type=="CNN_dropout":
		  self.model = CNN(image_size=image_size,hidden_size=N_units,droprates=droprates)
		elif "CNN_GFN" in self.model_type:
			self.model = CNN_MaskedDropout(image_size=image_size,hidden_size=N_units,droprates=droprates)

		elif self.model_type=="CNN_Standout":
			self.model = CNN_Standout(image_size=image_size,hidden_size=N_units,droprates=droprates)

		elif self.model_type=="CNN_SVD":
			self.model = CNN_SVD(image_size=image_size,hidden_size=N_units,droprates=droprates)


		####ReNet
		elif self.model_type=="RESNET_GFFN":#GFFN means faster version of the GFN
			self.model =Resenet_GFFN(image_size=image_size,
														in_dim=N_units,
														out_dim=10,
														hidden=(N_units,N_units,N_units,N_units),
														activation=nn.LeakyReLU,
														dropout_rate=droprates,
														mg_type='gfn',
														lr=1e-3,
														z_lr=1e-1,
														mg_lr=1e-3,
														mg_hidden=None,
														mg_activation=nn.LeakyReLU,
														beta=1,
														device=device,)


		elif self.model_type=="RESNET_dropoutAll":
			#dropour on all layers
		  self.model = Resenet_Alldrop(num_classes=10,image_size=image_size,hidden_size=N_units,droprates=droprates)


		self.model.to(device)
		self.criterion = nn.CrossEntropyLoss().to(device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
		
		self.loss_ = []
		self.GFN_losses=[]
		self.test_accuracy = []
		self.test_error = []
		self.test_accuracy_OOD = []
		self.test_error_OOD = []

		#####GFN flow function and samplling function 
		if "GFNFM" in self.model_type:
			self.GFN_operation=GFN_SamplingMask(N_units=N_units,batch_size=batch_size,device=device,p=droprates,Gamma=0.05)

			self.Fnet=FlowFunction(state_dim=N_units, n_action=N_units,condition_dim=N_units).to(device)

			self.optimizer_GFN = optim.Adam(self.Fnet.parameters(), lr=lr)
		
			total_params = sum(p.numel() for p in self.model.parameters())+sum(p.numel() for p in self.Fnet.parameters())

		elif "GFNDB" in self.model_type:
			self.GFN_operation=GFN_SamplingMask(N_units=N_units,batch_size=batch_size,device=device,p=droprates,Gamma=0.05)

			self.model_DB=DBModel(state_dim=N_units, n_action=N_units,condition_dim=N_units).to(device)

			self.optimizer_GFN = optim.Adam(self.model_DB.parameters(), lr=lr)
		
			total_params = sum(p.numel() for p in self.model.parameters())+sum(p.numel() for p in self.model_DB.parameters())

		else:
			total_params = sum(p.numel() for p in self.model.parameters())


		print("number of parameters:")
		print(total_params)


	def fit(self,verbose=True):
		if args.Data == "camelyon17": x_valid, y_valid, _ = iter(validloader).next()
		else:x_valid, y_valid = iter(validloader).next()

		####pick early stop, train mask etc.
		# x_valid, y_valid = iter(validloader).next()
		x_valid, y_valid = x_valid.to(device),y_valid.to(device)

		##augment x_valid once and use it for all epoch/step ( for reward type 2)
		x_valid_augmented=[]
		for idx in range(x_valid.shape[0]):
			####randomly augment half of the validation set
			if random.randrange(100)<50:
				augmenter=random.choice(augmenters_train)#randomly pick an augmenter
				vec_augmented=augmenter(x_valid[idx,:].unsqueeze(0))
				x_valid_augmented.append(vec_augmented)
			else:
				x_valid_augmented.append(x_valid[idx,:].unsqueeze(0))
		x_valid_augmented=torch.cat(x_valid_augmented,0)

		if args.Data == "camelyon17":
			x_test, y_test, _ = iter(testloader).next()
		else:
			x_test, y_test = iter(testloader).next()

		###Do not use test data to train mask or tune hyperparameter, it is cheating
		# x_test, y_test = iter(testloader).next()
		x_test,y_test = x_test.to(device),y_test.to(device)


		####test on augmented data ( different augmentations!)
		augmented_X_tests=[]
		
		for idx,augmenter in enumerate(augmenters_test):
				augmented_X_tests.append(augmenter(x_test.detach().clone().to(device)))

		best_valid_acc=0
		for epoch in range(self.max_epoch):
			
			running_loss = 0
			for i, data in enumerate(trainloader, 0):
				if args.Data == "camelyon17":
					inputs_, labels_, _ = data
				else:
					inputs_, labels_ = data

				# inputs_, labels_ = data
			
				inputs, labels = Variable(inputs_).to(device), Variable(labels_).to(device)
				self.optimizer.zero_grad()

				####augment images for GFN training 
				if "GFN" in self.model_type and args.RewardType==2:
					augmented_inputs=[]
					for idx, augmenter in enumerate(augmenters):
						if idx>=(len(augmenters)//2):
							augmented_inputs.append(augmenter(inputs_.detach().clone()).to(device))
					
				###forward
				if  "GFNFM" in self.model_type:
					
					self.GFN_operation.reset()

					condition=self.model.Get_condition(inputs).detach().clone().to(device)
					selected_=self.GFN_operation.forwardFM(FlowFunction=self.Fnet,conditions=condition)

					#selected_ = torch.zeros((inputs.shape[0],10)).uniform_(0,1)>0.5
					masks=selected_					
					outputs = self.model(inputs,masks)#####if using GFN to generate mask for dropout
					####augmentations
					if args.RewardType==2:
						with torch.no_grad():
							augmented_ouputs=[]
							for augmented_input in augmented_inputs:
								##original mask is use here, we want the same mask give good performance for difference augmentation						
								augmented_output = self.model(augmented_input,masks)#####if using GFN to generate mask for dropout

								augmented_ouputs.append(augmented_output.detach().clone())


				elif  "GFNDB" in self.model_type:

					self.GFN_operation.reset()

					###forwardDB will auotmatically reset the vectors for DB operations

					condition=self.model.Get_condition(inputs).detach().clone().to(device)

					selected_=self.GFN_operation.forwardDB(model_DB=self.model_DB,conditions=condition)

					#selected_ = torch.zeros((inputs.shape[0],10)).uniform_(0,1)>0.5
					masks=selected_
					outputs = self.model(inputs,masks)#####if using GFN to generate mask for dropout
					
					####augmentations
					if args.RewardType==2:
						with torch.no_grad():
							augmented_ouputs=[]
				
							for augmented_input in augmented_inputs:
								##original mask is use here, we want the same mask give good performance for difference augmentation						
								augmented_output = self.model(augmented_input,masks)#####if using GFN to generate mask for dropout

								augmented_ouputs.append(augmented_output.detach().clone())
						
				elif "GFFN" in self.model_type:
					bsz,n_channels,H,W=inputs.shape
					####this step update prediction parameters
					outputs,L_metric = self.model.step(x=inputs.reshape((bsz, -1)), y=labels )
					##this step update GFN parameters (gradient decent included)
					x_train=inputs.reshape((bsz, -1))
					y_train=labels


					if args.RewardType==0:
						G_metric = self.model._gfn_step(x_mask=x_train, y_mask=y_train ,x_reward=x_train, y_reward=y_train)
					elif args.RewardType==1:
						
						G_metric = self.model._gfn_step(x_mask=x_valid.reshape(x_valid.shape[0],-1), y_mask=y_valid ,x_reward=x_valid.reshape(x_valid.shape[0],-1), y_reward=y_valid)
					

					elif args.RewardType==2:
						##build mask using validation set but get reward from different augmentation of the validation set
						x_valid_original=x_valid
						y_valid_original=y_valid

						x_valid_original=x_valid_original.reshape(x_valid_original.shape[0],-1)
						x_valid_augmented=x_valid_augmented.reshape(x_valid_augmented.shape[0],-1)						
						# x_valid_augmented=[]
						# x_valid_original=[]
						# y_valid_original=[]
						# for idx, augmenter in enumerate(augmenters):
						# 	if idx>=(len(augmenters)//2):
						# 		x_valid_augmented.append(augmenter(x_valid).reshape(x_valid.shape[0],-1).detach().clone().to(device))
						# 		x_valid_original.append(x_valid.reshape(x_valid.shape[0],-1))
						# 		y_valid_original.append(y_valid)
						# x_valid_original=torch.cat(x_valid_original,0)
						# x_valid_augmented=torch.cat(x_valid_augmented,0)#augmented x_valid
						# y_valid_original=torch.cat(y_valid_original,0)

						G_metric = self.model._gfn_step(x_mask=x_valid_original, y_mask=y_valid_original ,x_reward=x_valid_augmented, y_reward=y_valid_original)
						
										

				else:
					outputs = self.model(inputs)

				if "GFFN" not in self.model_type:
					loss = self.criterion(outputs, labels)
					loss.backward()
					self.optimizer.step()
					running_loss += loss.item()

				else:
					loss=L_metric['loss']
					running_loss += loss
				
				#####update GFN

				if "GFN" in self.model_type:
					loss_batch = nn.CrossEntropyLoss(reduce=False)(outputs, labels)#rewards required for each element in the batch
					beta=args.beta ###higher parameter to tune sharpness of the peak
					if args.OODReward==0:
						rewards=torch.exp(-beta*loss_batch).detach().clone()
					elif args.OODReward==1:##consider losses augmentations from the training set

						loss_augmentation_batch=torch.zeros(loss_batch.shape).to(device)
						for augmented_output in augmented_ouputs:
							loss_augmentation_batch_ = nn.CrossEntropyLoss(reduce=False)(augmented_output, labels)#rewards required for each element in the batch
							loss_augmentation_batch=loss_augmentation_batch+loss_augmentation_batch_

						loss_augmentation_batch=loss_augmentation_batch/len(augmented_ouputs)
				
				
						rewards=torch.exp(-beta*(loss_batch+loss_augmentation_batch)).detach().clone()
						
				if "GFNFM" in self.model_type:
					self.optimizer_GFN.zero_grad()
					GFN_loss=self.GFN_operation.CalculateFlowMatchingLoss(self.Fnet,rewards,conditions=inputs.reshape(inputs.shape[0],-1).detach().clone())
					GFN_loss.backward()
					self.optimizer_GFN.step()

				if "GFNDB" in self.model_type:
					
					GFN_loss=self.GFN_operation.DB_train(rewards,self.optimizer_GFN)
				elif "GFFN" in self.model_type:
					GFN_loss=G_metric['tb_loss']
					

			self.loss_.append(running_loss / len(trainloader))
			if verbose and epoch%1==0:
				print('Epoch {} loss: {}'.format(epoch+1, self.loss_[-1]))
			
  
			if "GFN" in self.model_type:
				self.GFN_losses.append(GFN_loss.item())
			elif "GFFN" in self.model_type:
				self.GFN_losses.append(GFN_loss)
			else:
				self.GFN_losses.append(0)

			y_test_pred = self.predict(x_test).cpu()
			self.test_accuracy.append(np.mean((y_test.cpu() == y_test_pred.cpu()).numpy()))
			self.test_error.append(int(len(testset)*(1-self.test_accuracy[-1])))

			###validation acc
			y_valid_pred = self.predict(x_valid)

			valid_acc=np.mean((y_valid.cpu() == y_valid_pred.cpu()).numpy())

			####OOD loss
			OOD_accs=[]
			OOD_testerrors=[]
			for augmented_X_test in augmented_X_tests:
				y_test_pred = self.predict(augmented_X_test)
				OOD_accs.append(np.mean((y_test.cpu() == y_test_pred.cpu()).numpy()))
				OOD_testerrors.append((int(len(testset)*(1-self.test_accuracy[-1]))))

			
			self.test_accuracy_OOD.append(np.mean(OOD_accs))
			self.test_error_OOD.append(np.mean(OOD_testerrors))
				

			#df = pd.DataFrame({'train_loss':self.loss_,'test_acc':self.test_accuracy,'test_error':self.test_error,"GFN_loss":self.GFN_loss})
			
			df = pd.DataFrame({'train_loss':self.loss_,
							'test_acc':self.test_accuracy,
							'test_error':self.test_error,
							"GFN_losses":self.GFN_losses,
							'test_acc_OOD':self.test_accuracy_OOD,
							'test_error_OOD':self.test_error_OOD})
					


			df.to_csv("Results/"+Task_name+"_performance.csv")

			if verbose and epoch%1==0:
				print('Test error: {}; test accuracy: {} ,train loss: {} GFN loss: {} Valid_acc: {}'.format(self.test_error[-1], self.test_accuracy[-1],self.loss_[-1],self.GFN_losses[-1],valid_acc))
			
			if valid_acc> best_valid_acc:
				#use validation performance to decide early stopping
				torch.save(self.model.state_dict(), "checkpoints/"+Task_name+'.pt')
				best_valid_acc=valid_acc


		return self
	
	def predict(self, x):
		# Used to keep all test errors after each epoch
		model = self.model.eval()
		with torch.no_grad():
			if "GFNFM" in self.model_type:
				N_repeats=11
				outputs=[]
				for j in range(N_repeats):#repeat N times and sample the distribution
					self.GFN_operation.reset()
					self.GFN_operation.Train=False

					condition=self.model.Get_condition(x).detach().clone().to(device)
					
					selected_=self.GFN_operation.forwardFM(FlowFunction=self.Fnet,conditions=condition)

					#selected_ = torch.zeros((inputs.shape[0],10)).uniform_(0,1)>0.5
					mask=selected_
					outputs_vec = model(Variable(x),mask)#####if using GFN to generate mask for dropout
					
					outputs.append(outputs_vec.unsqueeze(0))
				outputs=torch.cat(outputs,0)
				
				outputs=outputs.mean(0)


			elif "GFNDB" in self.model_type:
				N_repeats=11
				outputs=[]
				for j in range(N_repeats):#repeat N times and sample the distribution
					self.GFN_operation.reset()
					self.GFN_operation.Train=False

					###forwardDB will auotmatically reset the vectors for DB operations
					
					
					condition=self.model.Get_condition(x).detach().clone().to(device)
					
					selected_=self.GFN_operation.forwardDB(model_DB=self.model_DB,conditions=condition)

					#selected_ = torch.zeros((inputs.shape[0],10)).uniform_(0,1)>0.5
					mask=selected_

					outputs_vec = model(Variable(x),mask)#####if using GFN to generate mask for dropout
					
					outputs.append(outputs_vec.unsqueeze(0))
				outputs=torch.cat(outputs,0)
				
				outputs=outputs.mean(0)

			elif "GFFN" in self.model_type:
				N_repeats=11
				outputs=[]
				for j in range(N_repeats):#repeat N times and sample the distribution
				
					bsz,n_channels,H,W=x.shape
					####this step already update GFN parameters
					outputs_vec=self.model(Variable(x).reshape((bsz, -1)))
					outputs.append(outputs_vec.unsqueeze(0))
				outputs=torch.cat(outputs,0)
				
				outputs=outputs.mean(0)



			else:
			  outputs = model(Variable(x))

			_, pred = torch.max(outputs.data, 1)
		model = self.model.train()
		return pred
	
	def __str__(self):
		return 'Hidden layers: {}; dropout rates: {}'.format(self.hidden_layers, self.droprates)




#######run the model 

## Below is training code, uncomment to train your own model... ###
### Note: You need GPU to run this section ###

# Define networks

if args.Data=="MNIST":
	#Input_size=28*28
	image_size=(1,28,28)
elif args.Data=="CIFAR10":
	#Input_size=3*32*32
	image_size=(3,32,32)
elif args.Data=="SVHN":
	#Input_size=3*32*32
	image_size=(3,32,32)
elif args.Data=="camelyon17":
	#Input_size=3*32*32
	image_size=(3,96,96)

mlp1 = [MLPClassifier(droprates=args.p,image_size=image_size,max_epoch=args.Epochs,model_type=args.Method,N_units=args.Hidden_dim)]

# mlp1 = [MLPClassifier(droprates=[0.0, 0.5], max_epoch=3,model_type="MLP"),
#         MLPClassifier(droprates=[0.0, 0.0], max_epoch=3,model_type="MLP_GFN"),
#         MLPClassifier(droprates=[0.0, 0.0], max_epoch=3,model_type="MLP_SVD"),
#        MLPClassifier(droprates=[0.0, 0.0], max_epoch=3,model_type="MLP_Standout")
#         ]
#       
# Training, set verbose=True to see loss after each epoch.
#[mlp.fit(trainset, testset,verbose=True) for mlp in mlp1]
[mlp.fit(verbose=True) for mlp in mlp1]


# # Save torch models
# for ind, mlp in enumerate(mlp1):
# 	#torch.save(mlp.model, 'mnist_mlp1_'+str(ind)+'.pth')
# 	torch.save(mlp.model, "checkpoints/"+Task_name+'.pth')
# 	# Prepare to save errors
# 	mlp.test_error = list(map(str, mlp.test_error))

# # Save test errors to plot figures
# open("Results/"+Task_name+"test_errors.txt","w").write('\n'.join([','.join(mlp.test_error) for mlp in mlp1])) 


# Load saved models to CPU
#mlp1_models = [torch.load('mnist_mlp1_'+str(ind)+'.pth',map_location={'cuda:0': 'cpu'}) for ind in [0,1,2]]

# Load saved test errors to plot figures.
# mlp1_test_errors = [error_array.split(',') for error_array in open("Results/"+Task_name+"test_errors.txt","r").read().split('\n')]
# mlp1_test_errors = np.array(mlp1_test_errors,dtype='f')




# #####visualization 


# labels = [args.Method] 
# #          'MLP 50% dropout in hidden layers',
#  #         'MLP 50% dropout in hidden layers+20% input layer']

# plt.figure(figsize=(8, 7))
# for i, r in enumerate(mlp1_test_errors):
# 	plt.plot(range(1, len(r)+1), r, '.-', label=labels[i], alpha=0.6);
# #plt.ylim([50, 250]);
# plt.legend(loc=1);
# plt.xlabel('Epochs');
# plt.ylabel('Number of errors in test set');
# plt.title('Test error on MNIST dataset for Multilayer Perceptron')
# plt.savefig('Results/'+Task_name+'PerformanceVsepisodes.png')
# plt.clf()
