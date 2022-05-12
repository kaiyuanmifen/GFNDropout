
import torch
import numpy as np
import torch.optim as optim 
from GFN_SampleMask import GFN_SamplingMask
from cifar10c import CIFAR_1O_Corrupted
from GFNFunctions import *
from Dropout_DIY import *
from TaskModels import *
import os

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
parser.add_argument('--folder', type=str, default='Results',
					help='Folder to store the results of the experiments')


args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)

EXP_FOLDER = args.folder
os.makedirs(EXP_FOLDER,exist_ok=True)


Task_name=args.Method+"_"+args.Data+"_"+str(args.Hidden_dim)+"_"+str(args.p)+"_"+str(args.beta)+"_"+str(args.RewardType)+"_"+str(args.DataRatio)+"_"+str(args.seed)
print("task:",Task_name)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####part 1 load MNIST/CIFAR data
batch_size=128
if args.Data=="MNIST":
	image_size_use = (224,224)
	mnist = datasets.MNIST(download=False, train=True, root="data/").data.float()
	#Transform for MNIST. Source: https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/ 
	transform = transforms.Compose([transforms.Resize(image_size_use),transforms.ToTensor(), transforms.Normalize(tuple([(mnist.mean()/255).numpy().tolist()]), tuple([(mnist.std()/255).numpy().tolist()]))])
	
	trainset = datasets.MNIST(root='data/', train=True, download=True, transform=transform)
	testset = datasets.MNIST(root='data/', train=False, transform=transform)

	indices = torch.randperm(len(trainset))[:int(len(trainset)*args.DataRatio)]

	validset =torch.utils.data.Subset(trainset, indices[int(0.7*len(indices)):(int(1*len(indices))-1)])
	
	trainset =torch.utils.data.Subset(trainset, indices[:int(0.7*len(indices))])

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

	validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
		

	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
	

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
				transforms.RandomResizedCrop(size=image_size_use),
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
	image_size_use = (224,224)
	cifar10 = torch.from_numpy(datasets.CIFAR10(download=False, train=True, root="data/").data).float()
	# CIFAR is of shape (BS,H,W,C)
	transform = transforms.Compose([transforms.Resize(image_size_use),transforms.ToTensor(), transforms.Normalize(tuple((cifar10.mean((0,1,2))/255).numpy().tolist()), tuple((cifar10.std((0,1,2))/255).numpy().tolist()))])
	
	#transform = transforms.Compose(
	#[transforms.ToTensor(),
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='data', train=True,
											download=True, transform=transform)


	testset = torchvision.datasets.CIFAR10(root='data', train=False,
										   download=True, transform=transform)
	
	#indices = torch.randperm(len(trainset))[:1000]
	#indices = torch.randperm(len(trainset))
	indices = torch.randperm(len(trainset))[:int(len(trainset)*args.DataRatio)]
	#indices = torch.randperm(len(trainset))

	validset =torch.utils.data.Subset(trainset, indices[int(0.7*len(indices)):(int(1*len(indices))-1)])
	
	trainset =torch.utils.data.Subset(trainset, indices[:int(0.7*len(indices))])
	#indices = torch.randperm(len(testset))[:300]
	indices = torch.randperm(len(testset))

	testset  =torch.utils.data.Subset(testset, indices)
	
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True)


	validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
		

	# Also use CIFAR 10 C for testing 
	CORRPUTED_FILES_DIR = '/home/mila/c/chris.emezue/GFNDropout/CIFAR-10-C'
	corrupted_cifar_test = CIFAR_1O_Corrupted(CORRPUTED_FILES_DIR,transform)
	corrupted_testloader = torch.utils.data.DataLoader(corrupted_cifar_test, batch_size=batch_size,
                        					shuffle=False)

	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
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
			transforms.RandomResizedCrop(size=image_size_use),
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
	image_size_use = (224,224)
	svhn = torch.from_numpy(datasets.SVHN(download=False, split="train", root="data/").data).float()
	# SVHN is of shape (BS,C,H,W)
	transform = transforms.Compose([transforms.Resize(image_size_use),transforms.ToTensor(), transforms.Normalize(tuple((svhn.mean((0,2,3))/255).numpy().tolist()), tuple((svhn.std((0,2,3))/255).numpy().tolist()))])

	
	#transform = transforms.Compose(
	#[transforms.ToTensor(),
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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


	validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
		


	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
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
			transforms.RandomResizedCrop(size=image_size_use),
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


#part 2 function to run the task

class MLPClassifier:
	def __init__(self, image_size,droprates=0.5, batch_size=128, max_epoch=10, \
				 lr=0.001, momentum=0,model_type="MLP_nodropout",N_units=50):
		# Wrap MLP model
		self.droprates = droprates
		self.batch_size = batch_size
		self.max_epoch = max_epoch
		self.model_type=model_type

		self.num_channels = image_size[0]
		self.num_resnet_layers = 101
		if self.model_type=="MLP_nodropout":
		  self.model = MLP(Input_size=image_size[0]*image_size[1]*image_size[2], hidden_size=N_units,droprates=0)
		
		elif self.model_type=="MLP_dropout":
		  self.model = MLP(Input_size=image_size[0]*image_size[1]*image_size[2], hidden_size=N_units,droprates=droprates)

		elif "MLP_GFN" in self.model_type:
		  self.model = MLP_MaskedDropout(Input_size=image_size[0]*image_size[1]*image_size[2],hidden_size=N_units)

		elif self.model_type=="MLP_SVD":
			self.model = MLP_SVD(Input_size=image_size[0]*image_size[1]*image_size[2],hidden_size=N_units)
		elif self.model_type=="MLP_Standout":
			self.model = MLP_Standout(Input_size=image_size[0]*image_size[1]*image_size[2],hidden_size=N_units,droprates=droprates)
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

		# ResNet
		elif self.model_type=="RESNET_nodropout":
			self.model = ResNet(num_layers=self.num_resnet_layers, img_channels=self.num_channels,hidden_size=N_units,droprates=0)
		elif self.model_type=="RESNET_dropout":
			self.model = ResNet(num_layers=self.num_resnet_layers, img_channels = self.num_channels,hidden_size=N_units,droprates=droprates)
		elif "RESNET_GFN" in self.model_type:
			self.model = ResNet_MaskedDropout(num_layers = self.num_resnet_layers, img_channels = self.num_channels,hidden_size=N_units,droprates=droprates)

		elif self.model_type=="RESNET_Standout":
			self.model = ResNet_Standout(num_layers = self.num_resnet_layers, img_channels = self.num_channels,hidden_size=N_units,droprates=droprates)

		elif self.model_type=="RESNET_SVD":
			self.model = ResNet_SVD(num_layers = self.num_resnet_layers, img_channels = self.num_channels,hidden_size=N_units,droprates=droprates)

		######Faster version of GFN
		elif self.model_type=="RESNET_GFFN":#GFFN means faster version of the GFN
			self.model =RESNETClassifierWithMaskGenerator(num_layers = self.num_resnet_layers,
														img_channels = self.num_channels,
														out_dim=10,
											            hidden=(N_units,N_units,N_units),
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
  
		else:
			raise Exception(f"Could not recognize model type `{self.model_type}` specified.")  	

		self.model.to(device)
		self.criterion = nn.CrossEntropyLoss().to(device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
		
		self.loss_ = []
		self.GFN_losses=[]
		self.test_accuracy = []
		self.test_error = []
		self.test_accuracy_OOD = []
		self.test_error_OOD = []
		self.corrupted_accs = []

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
		# Training, make sure it's on GPU, otherwise, very slow...

		

		best_valid_acc=0
		for epoch in range(self.max_epoch):
			running_loss = 0
			for i, data in enumerate(trainloader, 0):
		 
				inputs_, labels_ = data
			
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
					outputs,L_metric = self.model.step(x=inputs, y=labels )
					##this step update GFN parameters (gradient decent included)
					x_train=inputs
					y_train=labels


					if args.RewardType==0:
						G_metric = self.model._gfn_step(x_mask=x_train, y_mask=y_train ,x_reward=x_train, y_reward=y_train)
					elif args.RewardType==1:
						G_metric={}
						G_metric_batch_losses = []
						####pick early stop, train mask etc.
						for i, valid_data in enumerate(validloader):
							x_valid, y_valid = valid_data
							x_valid, y_valid = x_valid.to(device),y_valid.to(device)
							
										
							G_metric_batch = self.model._gfn_step(x_mask=x_valid, y_mask=y_valid ,x_reward=x_valid, y_reward=y_valid)
							G_metric_batch_losses.append(G_metric_batch['tb_loss'])
						G_metric['tb_loss'] = np.mean(G_metric_batch_losses)
					

					elif args.RewardType==2:
						##build mask using validation set but get reward from different augmentation of the validation set

						G_metric={}
						G_metric_batch_losses = []
						####pick early stop, train mask etc.
						for i, valid_data in enumerate(validloader):
							x_valid, y_valid = valid_data
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
							G_metric_batch = self.model._gfn_step(x_mask=x_valid, y_mask=y_valid ,x_reward=x_valid_augmented, y_reward=y_valid)
							G_metric_batch_losses.append(G_metric_batch['tb_loss'])
						G_metric['tb_loss'] = np.mean(G_metric_batch_losses)
						
										

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
					GFN_loss=self.GFN_operation.CalculateFlowMatchingLoss(self.Fnet,rewards,conditions=inputs.detach().clone())
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

			###Do not use test data to train mask or tune hyperparameter, it is cheating
			batch_test_accs = []
			batch_test_error = []
			OOD_accs = []
			OOD_testerrors = []

			for test_data in testloader:
				x_test, y_test = test_data
				x_test,y_test = x_test.to(device),y_test.to(device)
				y_test_pred = self.predict(x_test).cpu()
				acc_ = np.mean((y_test.cpu() == y_test_pred.cpu()).numpy())
				batch_test_accs.append(acc_)
				batch_test_error.append(int(len(x_test)*(1-acc_)))

				# do augmenting on test set and get metrics
				####test on augmented data ( different augmentations!)
				augmented_X_tests=[]
				OOD_batch_accs = []
				OOD_batch_error = []

				for idx,augmenter in enumerate(augmenters_test):
					augmented_X_tests.append(augmenter(x_test.detach().clone().to(device)))
				for augmented_X_test in augmented_X_tests:
					y_test_pred = self.predict(augmented_X_test)
					ood_acc_ = np.mean((y_test.cpu() == y_test_pred.cpu()).numpy())
					OOD_batch_accs.append(ood_acc_)
					OOD_batch_error.append(int(len(augmented_X_test)*(1-ood_acc_)))
				OOD_accs.append(np.mean(OOD_batch_accs))
				OOD_testerrors.append(np.mean(OOD_batch_error))
	
			self.test_accuracy.append(np.mean(batch_test_accs))
			self.test_error.append(np.mean(batch_test_error))

			###validation acc
			valid_acc_batch=[]
			for i, valid_data in enumerate(validloader):
				x_valid, y_valid = valid_data
				x_valid, y_valid = x_valid.to(device),y_valid.to(device)
				y_valid_pred = self.predict(x_valid)
				valid_acc_batch.append(np.mean((y_valid.cpu() == y_valid_pred.cpu()).numpy()))
			valid_acc = np.mean(valid_acc_batch)	


			self.test_accuracy_OOD.append(np.mean(OOD_accs))
			self.test_error_OOD.append(np.mean(OOD_testerrors))
				
			
			# Get accuracy on CIFAR 10 Corrupted 
			if args.Data=="CIFAR10":
				corrupted_accs = []
				for i, corrupted_test_data in enumerate(corrupted_testloader):
					c_inputs_, c_labels_ = corrupted_test_data
					c_inputs_, c_labels_ = Variable(c_inputs_).to(device), Variable(c_labels_).to(device)
					y_c_test_pred = self.predict(c_inputs_)
					corrupted_accs.append(np.mean((c_labels_.cpu() == y_c_test_pred.cpu()).numpy()))
		 
				
				self.corrupted_accs.append(np.mean(corrupted_accs))

			
				df = pd.DataFrame({'train_loss':self.loss_,
							'test_acc':self.test_accuracy,
							'test_error':self.test_error,
							"GFN_losses":self.GFN_losses,
							'test_acc_OOD':self.test_accuracy_OOD,
							'test_error_OOD':self.test_error_OOD,
							'CIFAR_10C_acc':self.corrupted_accs})
			else:
				df = pd.DataFrame({'train_loss':self.loss_,
							'test_acc':self.test_accuracy,
							'test_error':self.test_error,
							"GFN_losses":self.GFN_losses,
							'test_acc_OOD':self.test_accuracy_OOD,
							'test_error_OOD':self.test_error_OOD})
							


			df.to_csv(f"{EXP_FOLDER}/"+Task_name+"_performance.csv")

			if verbose and epoch%1==0:
				print('Test error: {}; test accuracy: {} ,train loss: {} GFN loss: {} Valid_acc: {}'.format(self.test_error[-1], self.test_accuracy[-1],self.loss_[-1],self.GFN_losses[-1],valid_acc))
			
			if valid_acc> best_valid_acc:
				#use validation performance to decide early stopping
				torch.save(self.model.state_dict(), "/home/mila/c/chris.emezue/scratch/gfndropout/checkpoints/"+Task_name+'.pt')
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
				
						#bsz,n_channels,H,W=x.shape
						####this step already update GFN parameters
						outputs_vec=self.model(Variable(x))
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
	#image_size=(1,28,28)
	image_size=(1,image_size_use[0],image_size_use[1])
elif args.Data=="CIFAR10":
	#Input_size=3*32*32
	#image_size=(3,32,32)
	image_size=(3,image_size_use[0],image_size_use[1])
elif args.Data=="SVHN":
	#Input_size=3*32*32
	#image_size=(3,32,32)
	image_size=(3,image_size_use[0],image_size_use[1])

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

# Save torch models
for ind, mlp in enumerate(mlp1):
	#torch.save(mlp.model, 'mnist_mlp1_'+str(ind)+'.pth')

	#Chrs Emezue: saving in scratch due to file size.
	torch.save(mlp.model, "/home/mila/c/chris.emezue/scratch/gfndropout/checkpoints/"+Task_name+'.pth')
	# Prepare to save errors
	mlp.test_error = list(map(str, mlp.test_error))

# Save test errors to plot figures
#open(f"{EXP_FOLDER}/"+Task_name+"test_errors.txt","w").write('\n'.join([','.join(mlp.test_error) for mlp in mlp1])) 


# Load saved models to CPU
#mlp1_models = [torch.load('mnist_mlp1_'+str(ind)+'.pth',map_location={'cuda:0': 'cpu'}) for ind in [0,1,2]]

# Load saved test errors to plot figures.
#mlp1_test_errors = [error_array.split(',') for error_array in open(f"{EXP_FOLDER}/"+Task_name+"test_errors.txt","r").read().split('\n')]
#mlp1_test_errors = np.array(mlp1_test_errors,dtype='f')




#####visualization 


#labels = [args.Method] 
#          'MLP 50% dropout in hidden layers',
 #         'MLP 50% dropout in hidden layers+20% input layer']

#plt.figure(figsize=(8, 7))
#for i, r in enumerate(mlp1_test_errors):
#	plt.plot(range(1, len(r)+1), r, '.-', label=labels[i], alpha=0.6);
#plt.ylim([50, 250]);
#plt.legend(loc=1);
#plt.xlabel('Epochs');
#plt.ylabel('Number of errors in test set');
#plt.title('Test error on MNIST dataset for RESNET')
#plt.savefig(f"{EXP_FOLDER}/"+Task_name+'PerformanceVsepisodes.png')
#plt.clf()
