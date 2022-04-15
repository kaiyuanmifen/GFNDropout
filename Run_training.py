
import torch
import numpy as np
import torch.optim as optim 
from GFN_SampleMask import GFN_SamplingMask

from FlowFunction import FlowFunction
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

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)


Task_name=args.Method+"_"+args.Data+"_"+str(args.Hidden_dim)+"_"+str(args.p)+"_"+str(args.seed)

print("task:",Task_name)







device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####part 1 load MNIST/CIFAR data
batch_size=128
if args.Data=="MNIST":

	transform = transforms.Compose([transforms.ToTensor(), \
									transforms.Normalize((0), (1))])

	trainset = datasets.MNIST(root='data/', train=True, download=True, transform=transform)
	testset = datasets.MNIST(root='data/', train=False, transform=transform)

	indices = torch.randperm(len(trainset))[:1000]
	#indices = torch.randperm(len(trainset))

	trainset =torch.utils.data.Subset(trainset, indices)

	indices = torch.randperm(len(testset))[:1000]
	indices = torch.randperm(len(testset))

	testset  =torch.utils.data.Subset(testset, indices)

	print("training set length")
	print(len(trainset))

	print("test set length")
	print(len(testset))

	# Visualize 10 image samples in MNIST dataset
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
	dataiter = iter(trainloader)
	images, labels = dataiter.next()
	# # plot 10 sample images
	_,ax = plt.subplots(1,10)
	ax = ax.flatten()
	iml = images[0].numpy().shape[1]
	[ax[i].imshow(np.transpose(images[i].numpy(),(1,2,0)).reshape(iml,-1),cmap='Greys') for i in range(10)]
	[ax[i].set_axis_off() for i in range(10)]
	plt.savefig('DataExamples.png')

	print('label:',labels[:10].numpy())
	print('image data shape:',images[0].numpy().shape)

if args.Data=="CIFAR10":

	transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
											download=True, transform=transform)


	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
										   download=True, transform=transform)
	
	indices = torch.randperm(len(trainset))[:1000]
	#indices = torch.randperm(len(trainset))

	trainset =torch.utils.data.Subset(trainset, indices)

	indices = torch.randperm(len(testset))[:1000]
	indices = torch.randperm(len(testset))

	testset  =torch.utils.data.Subset(testset, indices)
	
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True, num_workers=2)

	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											 shuffle=False, num_workers=2)

	print("training set length")
	print(len(trainset))

	print("test set length")
	print(len(testset))

	classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	def imshow(img):
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('DataExamplesCIFAR.png')

	####show some examples
	dataiter = iter(trainloader)
	images, labels = dataiter.next()
	print("images")
	print(images.shape)

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


#part 2 function to run the task

class MLPClassifier:
	def __init__(self, Input_size,droprates=0.5, batch_size=128, max_epoch=10, \
				 lr=0.001, momentum=0,model_type="MLP",N_units=50):
		# Wrap MLP model
		self.droprates = droprates
		self.batch_size = batch_size
		self.max_epoch = max_epoch
		self.model_type=model_type

		if self.model_type=="MLP":
		  self.model = MLP(Input_size=Input_size, hidden_size=N_units,droprates=droprates)
		elif self.model_type=="CNN":
		  self.model = CNN(droprates)
		elif self.model_type=="MLP_GFN":
		  self.model = MLP_MaskedDropout(Input_size=Input_size,hidden_size=N_units)

		elif self.model_type=="MLP_SVD":
			self.model = MLP_SVD(Input_size=Input_size,hidden_size=N_units)
		elif self.model_type=="MLP_Standout":
			self.model = MLP_Standout(Input_size=Input_size,hidden_size=N_units,droprates=droprates)




		self.model.to(device)
		self.criterion = nn.CrossEntropyLoss().to(device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
		
		self.loss_ = []
		self.GFN_loss=[]
		self.test_accuracy = []
		self.test_error = []

		#####GFN flow function and samplling function 
				
		self.GFN_operation=GFN_SamplingMask(N_units=N_units,batch_size=batch_size,device=device,p=droprates,Gamma=0.05)

		self.Fnet=FlowFunction(state_dim=N_units, n_action=N_units,condition_dim=Input_size).to(device)

		self.optimizer_GFN = optim.Adam(self.Fnet.parameters(), lr=lr)
		
		if "GFN" in self.model_type:
			total_params = sum(p.numel() for p in self.model.parameters())+sum(p.numel() for p in self.model.parameters())+sum(p.numel() for p in self.Fnet.parameters())+sum(p.numel() for p in self.Fnet.parameters())

		else:
			total_params = sum(p.numel() for p in self.model.parameters())

		print("number of parameters:")
		print(total_params)


	def fit(self, trainset, testset, verbose=True):
		# Training, make sure it's on GPU, otherwise, very slow...
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
		X_test, y_test = iter(testloader).next()
		X_test = X_test.to(device)
		for epoch in range(self.max_epoch):
			
			running_loss = 0
			for i, data in enumerate(trainloader, 0):
		 
				inputs, labels = data
				inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
				self.optimizer.zero_grad()

			   
				if  self.model_type=="MLP_GFN":
					
					self.GFN_operation.reset()
					selected_=self.GFN_operation.forward(FlowFunction=self.Fnet,conditions=inputs.reshape(inputs.shape[0],-1))

					#selected_ = torch.zeros((inputs.shape[0],10)).uniform_(0,1)>0.5
					masks=[selected_,selected_]
					outputs = self.model(inputs,masks)#####if using GFN to generate mask for dropout
					
				else:
					outputs = self.model(inputs)

				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()
				running_loss += loss.item()

				#####matching loss to update GFN
				if self.model_type=="MLP_GFN":
					self.optimizer_GFN.zero_grad()
					loss_batch = nn.CrossEntropyLoss(reduce=False)(outputs, labels)#rewards required for each element in the batch
					beta=0.1 ###higher parameter to tune sharpness of the peak
					rewards=torch.exp(-beta*loss_batch).detach().clone()
		
					Match_loss=self.GFN_operation.CalculateFlowMatchingLoss(self.Fnet,rewards,conditions=inputs.reshape(inputs.shape[0],-1).detach().clone())
					Match_loss.backward()
					self.optimizer_GFN.step()


			self.loss_.append(running_loss / len(trainloader))
			if verbose and epoch%1==0:
				print('Epoch {} loss: {}'.format(epoch+1, self.loss_[-1]))
			y_test_pred = self.predict(X_test).cpu()
  
			if self.model_type=="MLP_GFN":
				self.GFN_loss.append(Match_loss.item())
			else:
				self.GFN_loss.append(0)
			
			self.test_accuracy.append(np.mean((y_test == y_test_pred).numpy()))
			self.test_error.append(int(len(testset)*(1-self.test_accuracy[-1])))
			
			#df = pd.DataFrame({'train_loss':self.loss_,'test_acc':self.test_accuracy,'test_error':self.test_error,"GFN_loss":self.GFN_loss})
			
			df = pd.DataFrame({'train_loss':self.loss_,'test_acc':self.test_accuracy,'test_error':self.test_error})
						


			df.to_csv("Results/"+Task_name+"_performance.csv")

			if verbose and epoch%1==0:
				print('Test error: {}; test accuracy: {}'.format(self.test_error[-1], self.test_accuracy[-1]))
		return self
	
	def predict(self, x):
		# Used to keep all test errors after each epoch
		model = self.model.eval()
		with torch.no_grad():
			if self.model_type=="MLP_GFN":
				N_repeats=11
				outputs=[]
				for j in range(N_repeats):#repeat N times and sample the distribution
					self.GFN_operation.reset()
					self.GFN_operation.Train=False
					selected_=self.GFN_operation.forward(FlowFunction=self.Fnet,conditions=x.reshape(x.shape[0],-1))

					#selected_ = torch.zeros((inputs.shape[0],10)).uniform_(0,1)>0.5
					masks=[selected_,selected_]
					outputs_vec = model(Variable(x),masks)#####if using GFN to generate mask for dropout
					
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
	Input_size=28*28
elif args.Data=="CIFAR10":
	Input_size=3*32*32

mlp1 = [MLPClassifier(droprates=args.p,Input_size=Input_size,max_epoch=args.Epochs,model_type=args.Method,N_units=args.Hidden_dim)]

# mlp1 = [MLPClassifier(droprates=[0.0, 0.5], max_epoch=3,model_type="MLP"),
#         MLPClassifier(droprates=[0.0, 0.0], max_epoch=3,model_type="MLP_GFN"),
#         MLPClassifier(droprates=[0.0, 0.0], max_epoch=3,model_type="MLP_SVD"),
#        MLPClassifier(droprates=[0.0, 0.0], max_epoch=3,model_type="MLP_Standout")
#         ]
#       
# Training, set verbose=True to see loss after each epoch.
[mlp.fit(trainset, testset,verbose=True) for mlp in mlp1]

# Save torch models
for ind, mlp in enumerate(mlp1):
	#torch.save(mlp.model, 'mnist_mlp1_'+str(ind)+'.pth')
	torch.save(mlp.model, "checkpoints/"+Task_name+'.pth')
	# Prepare to save errors
	mlp.test_error = list(map(str, mlp.test_error))

# Save test errors to plot figures
open("Results/"+Task_name+"test_errors.txt","w").write('\n'.join([','.join(mlp.test_error) for mlp in mlp1])) 


# Load saved models to CPU
#mlp1_models = [torch.load('mnist_mlp1_'+str(ind)+'.pth',map_location={'cuda:0': 'cpu'}) for ind in [0,1,2]]

# Load saved test errors to plot figures.
mlp1_test_errors = [error_array.split(',') for error_array in open("Results/"+Task_name+"test_errors.txt","r").read().split('\n')]
mlp1_test_errors = np.array(mlp1_test_errors,dtype='f')




#####visualization 


labels = [args.Method] 
#          'MLP 50% dropout in hidden layers',
 #         'MLP 50% dropout in hidden layers+20% input layer']

plt.figure(figsize=(8, 7))
for i, r in enumerate(mlp1_test_errors):
	plt.plot(range(1, len(r)+1), r, '.-', label=labels[i], alpha=0.6);
#plt.ylim([50, 250]);
plt.legend(loc=1);
plt.xlabel('Epochs');
plt.ylabel('Number of errors in test set');
plt.title('Test error on MNIST dataset for Multilayer Perceptron')
plt.savefig('Results/'+Task_name+'PerformanceVsepisodes.png')
plt.clf()
