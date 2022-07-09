import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torchvision import models, datasets, transforms
import time
from torch import nn
import pandas as pd


import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_CIFAR10(root="../../../data/"):
	input_size = 32
	num_classes = 10
	normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	
	# train_transform = transforms.Compose(
	# 		[
	# 		transforms.Resize(256), transforms.CenterCrop(224),
	# 		transforms.ToTensor(),
	# 		normalize,
	# 	]
	# )

	train_transform = transforms.Compose(
		[
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]
	)
	train_dataset = datasets.CIFAR10(
		root, train=True, transform=train_transform, download=True
	)

	test_transform = transforms.Compose(
		[
			transforms.ToTensor(),
			normalize,
		]
	)
	test_dataset = datasets.CIFAR10(
		root, train=False, transform=test_transform, download=True
	)

	return input_size, num_classes, train_dataset, test_dataset



def get_CIFAR100(root="../../../data/"):
	input_size = 32
	num_classes = 100
	normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	
	# train_transform = transforms.Compose(
	# 		[
	# 		transforms.Resize(256), transforms.CenterCrop(224),
	# 		transforms.ToTensor(),
	# 		normalize,
	# 	]
	# )

	train_transform = transforms.Compose(
		[
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]
	)
	train_dataset = datasets.CIFAR100(
		root, train=True, transform=train_transform, download=True
	)

	test_transform = transforms.Compose(
		[
			transforms.ToTensor(),
			normalize,
		]
	)
	test_dataset = datasets.CIFAR100(
		root, train=False, transform=test_transform, download=True
	)

	return input_size, num_classes, train_dataset, test_dataset



def get_SVHN(root="../../../data/"):
	input_size = 32
	num_classes = 10
	normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	
	# train_transform = transforms.Compose(
	# 		[
	# 		transforms.Resize(256), transforms.CenterCrop(224),
	# 		transforms.ToTensor(),
	# 		normalize,
	# 	]
	# )

	train_transform = transforms.Compose(
		[
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]
	)
	train_dataset = datasets.SVHN(
		root, split="train", transform=train_transform, download=True
	)

	test_transform = transforms.Compose(
		[
			transforms.ToTensor(),
			normalize,
		]
	)
	test_dataset = datasets.SVHN(
		root, split="test", transform=test_transform, download=True
	)

	return input_size, num_classes, train_dataset, test_dataset



class Model(torch.nn.Module):
	def __init__(self,args,num_classes):
		super().__init__()

		if args.pretrained:
			self.resnet = models.resnet18(pretrained=True, num_classes=1000)#pretrained model got 1000 classes (trained in imageNet)
			self.resnet.fc = nn.Linear(512, num_classes)
			#self.resnet.fc = MLP_GFFN(512, 10)#a 3 layer MLP
			
		else:
			self.resnet = models.resnet18(pretrained=pretrained, num_classes=num_classes)

			self.resnet.conv1 = torch.nn.Conv2d(
			3, 64, kernel_size=3, stride=1, padding=1, bias=False
			)

		self.resnet.maxpool = torch.nn.Identity()



		###if freeze the ResNet 
		if args.HowToTrain=="LinearProbing":
			for param in self.resnet.parameters():
				param.requires_grad = False

			self.resnet.fc.requires_grad_(True)


		print("ResNet trainable parameters")
		for name, param in self.resnet.named_parameters():
			if param.requires_grad:
				print (name)

		####GFN related
		# maskgenerator_input_shapes=[(64,16,16),(64,16,16),
		# 							(128,8,8),(128,8,8),
		# 							(256,4,4),(256,4,4),
		# 							(512,2,2),(512,2,2)]

		maskgenerator_input_shapes=[(512,2,2),(512,2,2)]

		#maskgenerator_input_shapes=[(512,2,2),(512,2,2)]

		#hiddens=[1350,1350,294,294,54,54,6,6]

		# self.rand_mask_generators=construct_random_mask_generators(n_layers=len(maskgenerator_input_shapes),
		# 															dropout_rate=0.5)
		
		if args.Dropout=="GFN":
			self.mask_generators=construct_mlp_mask_generators(input_shapes=maskgenerator_input_shapes,
															dropout_rate=None)
		
		else:
			self.mask_generators=construct_random_mask_generators(n_layers=len(maskgenerator_input_shapes),
																	dropout_rate=0.5)
		
		self.activation=nn.ReLU
		mg_activation=nn.LeakyReLU
		z_lr=1e-1
		mg_lr=1e-2
		self.total_flowestimator = MLP(in_dim=3*32*32, out_dim=1,
											activation=nn.LeakyReLU)

		MaskGeneratorParameters=[]
		for generator in self.mask_generators:
			MaskGeneratorParameters+=list(generator.parameters())

		param_list = [{'params': MaskGeneratorParameters, 'lr': mg_lr},
					 {'params': self.total_flowestimator.parameters(), 'lr': z_lr}]
		

		self.mg_optimizer = optim.Adam(param_list)



	def forward(self, x):
		x = self.resnet(x)
		#x = F.log_softmax(x, dim=1)

		return x

	def GFFN_forward(self, x):
		
		all_ims=[]
		masks=[]
		#intial block
		x = self.resnet.conv1(x)
		x=self.resnet.bn1(x)
		x=self.resnet.relu(x)
		x=self.resnet.maxpool(x)

		### layer 1 block1
		identity=x
		out=self.resnet.layer1[0].conv1(x)
		out=self.resnet.layer1[0].bn1(out)
		out=self.resnet.layer1[0].relu(out)

		out=self.resnet.layer1[0].conv2(out)
		out=self.resnet.layer1[0].bn2(out)

		if self.resnet.layer1[0].downsample is not None:
			identity = resnet.layer1[0].downsample(x)

		# intermediate_state=out.detach().clone()
		# all_ims.append(intermediate_state)
		# m=self.mask_generators[0](out).detach().unsqueeze(2).unsqueeze(3)

		# masks.append(m)
		# out=out.mul(m) 

		out += identity
		out = self.resnet.layer1[0].relu(out)
		x=out
		
		###layer 1 block 2

		identity=x
		out=self.resnet.layer1[1].conv1(x)
		out=self.resnet.layer1[1].bn1(out)
		out=self.resnet.layer1[1].relu(out)

		out=self.resnet.layer1[1].conv2(out)
		out=self.resnet.layer1[1].bn2(out)

		if self.resnet.layer1[1].downsample is not None:
			identity = resnet.layer1[1].downsample(x)



		# intermediate_state=out.detach().clone()
		# all_ims.append(intermediate_state)
		# m=self.mask_generators[1](out).detach().unsqueeze(2).unsqueeze(3)
		# masks.append(m)
		# out=out.mul(m) 
		

		out += identity
		out = self.resnet.layer1[1].relu(out)
		x=out
		

		##layer 2 block 1

		identity=x
		out=self.resnet.layer2[0].conv1(x)
		out=self.resnet.layer2[0].bn1(out)
		out=self.resnet.layer2[0].relu(out)



		
		out=self.resnet.layer2[0].conv2(out)
		out=self.resnet.layer2[0].bn2(out)

		if self.resnet.layer2[0].downsample is not None:
			identity = self.resnet.layer2[0].downsample(x)

		#print("layer2 block 1")
		#print(out.shape)
		# intermediate_state=out.detach().clone()
		# all_ims.append(intermediate_state)
		# m=self.mask_generators[2](out).detach().unsqueeze(2).unsqueeze(3)
		# masks.append(m)
		# out=out.mul(m) 

		out += identity
		out = self.resnet.layer2[0].relu(out)
		x=out


		##layer 2 block 2

		identity=x
		out=self.resnet.layer2[1].conv1(x)
		out=self.resnet.layer2[1].bn1(out)
		out=self.resnet.layer2[1].relu(out)


		
		out=self.resnet.layer2[1].conv2(out)
		out=self.resnet.layer2[1].bn2(out)

		if self.resnet.layer2[1].downsample is not None:
			identity = resnet.layer2[1].downsample(x)

		#print("layer2 block 2")
		#print(out.shape)
		# intermediate_state=out.detach().clone()
		# all_ims.append(intermediate_state)
		# m=self.mask_generators[3](out).detach().unsqueeze(2).unsqueeze(3)
		# masks.append(m)
		# out=out.mul(m) 

		out += identity
		out = self.resnet.layer2[1].relu(out)
		x=out		



		##layer 3 block 1

		identity=x
		out=self.resnet.layer3[0].conv1(x)
		out=self.resnet.layer3[0].bn1(out)
		out=self.resnet.layer3[0].relu(out)


		
		out=self.resnet.layer3[0].conv2(out)
		out=self.resnet.layer3[0].bn2(out)

		if self.resnet.layer3[0].downsample is not None:
			identity = self.resnet.layer3[0].downsample(x)

		#print("layer3 block 1")
		#print(out.shape)
		# intermediate_state=out.detach().clone()
		# all_ims.append(intermediate_state)
		# m=self.mask_generators[0](out).detach().unsqueeze(2).unsqueeze(3)
		# masks.append(m)
		# out=out.mul(m) 
		

		out += identity
		out = self.resnet.layer3[0].relu(out)
		x=out


		##layer 3 block 2

		identity=x
		out=self.resnet.layer3[1].conv1(x)
		out=self.resnet.layer3[1].bn1(out)
		out=self.resnet.layer3[1].relu(out)


		out=self.resnet.layer3[1].conv2(out)
		out=self.resnet.layer3[1].bn2(out)

		if self.resnet.layer3[1].downsample is not None:
			identity = resnet.layer3[1].downsample(x)

		#print("layer3 block 2")
		#print(out.shape)
		# intermediate_state=out.detach().clone()
		# all_ims.append(intermediate_state)
		# m=self.mask_generators[1](out).detach().unsqueeze(2).unsqueeze(3)
		# masks.append(m)
		# out=out.mul(m) 
		

		out += identity
		out = self.resnet.layer3[1].relu(out)
		x=out		



		##layer 4 block 1

		identity=x
		out=self.resnet.layer4[0].conv1(x)
		out=self.resnet.layer4[0].bn1(out)
		out=self.resnet.layer4[0].relu(out)


		#print("layer4 block 1")
		#print(out.shape)

		
		out=self.resnet.layer4[0].conv2(out)
		out=self.resnet.layer4[0].bn2(out)

		if self.resnet.layer4[0].downsample is not None:
			identity = self.resnet.layer4[0].downsample(x)


		intermediate_state=out.detach().clone()
		all_ims.append(intermediate_state)
		m=self.mask_generators[0](out).detach().unsqueeze(2).unsqueeze(3)
		masks.append(m)
		out=out.mul(m) 


		out += identity
		out = self.resnet.layer4[0].relu(out)



		x=out




		##layer 4 block 2

		identity=x
		out=self.resnet.layer4[1].conv1(x)
		out=self.resnet.layer4[1].bn1(out)
		out=self.resnet.layer4[1].relu(out)

		# print("layer4 block 2")
		# print(out.shape)


		
		out=self.resnet.layer4[1].conv2(out)
		out=self.resnet.layer4[1].bn2(out)

		if self.resnet.layer4[1].downsample is not None:
			identity = resnet.layer4[1].downsample(x)



		intermediate_state=out.detach().clone()
		all_ims.append(intermediate_state)
		m=self.mask_generators[1](out).detach().unsqueeze(2).unsqueeze(3)
		masks.append(m)
		out=out.mul(m)

		out += identity
		out = self.resnet.layer4[1].relu(out)


		
		x=out		



		####fc layer 

		x=self.resnet.avgpool(x)
		x = torch.flatten(x, 1)
		x=self.resnet.fc(x)
		#x = F.log_softmax(x, dim=1)	
			
	
		return x,masks,all_ims




	def _gfn_step(self, x_mask, y_mask,x_reward=None,y_reward=None):
		#####this step allows us to use different x,y to generate mask and calcualte reward(loss)

		metric = {}
	 
		if x_reward!=None:
			#generate mask
			_, masks,intermediate_states = self.GFFN_forward(x_mask)

			###for loss
			logits, _,_ = self.GFFN_forward_predefinedmasks(x_reward, masks)
		
		else:
			logits, masks,intermediate_states = self.GFFN_forward(x_mask)
			x_reward=x_mask
			y_reward=y_mask 


		beta=0.1#temperature
		with torch.no_grad():
			losses = nn.CrossEntropyLoss(reduce=False)(logits, y_reward)
			#losses =torch.nn.NLLLoss(reduce=False)(logits, y_reward)
			log_rewards = - beta * losses
			logZ=self.total_flowestimator(x_mask.reshape(x_mask.shape[0],-1))#this flow is calculated using x_mask, not a bug , to encourage generalization 
		# trajectory balance loss
		log_probs_F = []
		log_probs_B = []



		for m, mg_f,ims in zip(masks, self.mask_generators,intermediate_states):
			m=m.squeeze(3).squeeze(2)
	
			log_probs_F.append(mg_f.log_prob(ims, m).unsqueeze(1))
			#log_probs_B.append(mg_b.log_prob(m, m).unsqueeze(1))
			log_probs_B.append(torch.zeros(m.shape[0],1).to(device))
					
	
		tb_loss = ((logZ - log_rewards
					+ torch.cat(log_probs_F, dim=1).sum(dim=1)
					- torch.cat(log_probs_B, dim=1).sum(dim=1)) ** 2).mean()
		metric['tb_loss'] = tb_loss.item()
		self.mg_optimizer.zero_grad()
		tb_loss.backward()
		self.mg_optimizer.step()


		###calculated actual droppout rate
		actual_dropout_rate=0
		for m in masks:
			actual_dropout_rate+=m.sum(1).mean(0)/(m.shape[1]) 
		metric['actual_dropout_rate']=(actual_dropout_rate/len(masks)).item()

		return metric['tb_loss'],metric['actual_dropout_rate']


def train(model, train_loader, optimizer, epoch,args):
	model.train()


	total_loss = []
	total_GFNLoss=[]
	actual_dropout_rates=[]
	for data, target in tqdm(train_loader):
		data = data.to(device)
		target = target.to(device)

		optimizer.zero_grad()


		if args.Dropout in ["GFN","Random"]:
			logits,masks,intermediate_states=model.GFFN_forward(data)

		elif args.Dropout=="None":
			logits = model(data)


		#loss = F.nll_loss(prediction, target)
		loss=nn.CrossEntropyLoss()(logits,target)

		loss.backward()
		optimizer.step()

		total_loss.append(loss.item())

		if args.Dropout=="GFN":
			GFN_loss,actual_dropout_rate=model._gfn_step(data,target)
			total_GFNLoss.append(GFN_loss)
			actual_dropout_rates.append(actual_dropout_rate)
			

	avg_loss = sum(total_loss) / len(total_loss)
	print(f"Epoch: {epoch}:")
	if args.Dropout=="GFN":
		avg_GFNloss = sum(total_GFNLoss) / len(total_GFNLoss)
		avg_actual_dropout_rates=sum(actual_dropout_rates)/len(actual_dropout_rates)
		print(f"Train Set: Average Loss: {avg_loss:.2f} Average GFNLoss: {avg_GFNloss:.2f} average dropout rates: {avg_actual_dropout_rates:.4f}")
		
		return avg_loss,avg_GFNloss,avg_actual_dropout_rates

	else:
		print(f"Train Set: Average Loss: {avg_loss:.2f}")
		return avg_loss,0,0

def test(model, test_loader,args):
	model.eval()

	loss = 0
	correct = 0

	for data, target in test_loader:
		with torch.no_grad():
			data = data.to(device)
			target = target.to(device)


			if args.Dropout in ["GFN","Random"]:
				prediction=[]
				all_masks=[]
				for _ in range(args.SampleRepeats):
					prediction_,masks_,_=model.GFFN_forward(data)
					prediction.append(prediction_.unsqueeze(2))
					masks_=torch.cat(masks_,1)
					masks_=masks_.reshape(masks_.shape[0],masks_.shape[1]).unsqueeze(2)
					all_masks.append(masks_)
				all_masks=torch.cat(all_masks,2)#(bsz,N_units,repeats),only save mask of one batch
			
				prediction=torch.exp(torch.cat(prediction,2))
				prediction=torch.log(prediction.mean(2)) ##be careful, this mean needs to be in p space (0-1) not log space


			elif args.Dropout=='None':
				prediction = model(data)
				all_masks=[]
			#loss += F.nll_loss(prediction, target, reduction="sum")
			loss += nn.CrossEntropyLoss(reduction="sum")(prediction, target)

			prediction = prediction.max(1)[1]
			correct += prediction.eq(target.view_as(prediction)).sum().item()

	loss /= len(test_loader.dataset)

	percentage_correct = 100.0 * correct / len(test_loader.dataset)

	print(
		"Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
			loss, correct, len(test_loader.dataset), percentage_correct
		)
	)

	return loss, percentage_correct,all_masks


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--epochs", type=int, default=1000, help="number of epochs to train (default: 50)"
	)
	parser.add_argument(
		"--lr", type=float, default=0.05, help="learning rate (default: 0.05)"
	)
	parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

	parser.add_argument("--pretrained", type=int, default=1, help="whether to use pretrained model (default: 1)")


	parser.add_argument("--HowToTrain", type=str, default="LinearProbing", help="How to train the model (default: LinearProbing)")

	parser.add_argument("--Dropout", type=str, default="GFN", help="How to dropout (default: GFN)")

	parser.add_argument("--SampleRepeats", type=int, default=5, help="Number of repeats to sample masks during inference")

	parser.add_argument("--Name", type=str, default="", help="additional name of the experiment")



	parser.add_argument("--Data", type=str, default="", help="what data to use")



	args = parser.parse_args()
	print(args)
	args.pretrained=args.pretrained==1

	torch.manual_seed(args.seed)

	if args.Data=="CIFAR10":
		input_size, num_classes, train_dataset, test_dataset = get_CIFAR10()
	elif args.Data=="CIFAR100":
		input_size, num_classes, train_dataset, test_dataset = get_CIFAR100()
	elif args.Data=="SVHN":
		input_size, num_classes, train_dataset, test_dataset = get_SVHN()



	kwargs = {"num_workers": 2, "pin_memory": True}

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=128, shuffle=True, **kwargs
	)
	test_loader = torch.utils.data.DataLoader(
		test_dataset, batch_size=5000, shuffle=False, **kwargs
	)

	model = Model(args,num_classes)
	model = model.to(device)
	



	milestones = [25, 40]

	optimizer = torch.optim.SGD(
		model.resnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
	)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(
		optimizer, milestones=milestones, gamma=0.1
	)

	model_name=args.Data+"_"+str(args.pretrained)+"_"+str(args.pretrained)+"_"+str(args.HowToTrain)+"_"+args.Dropout+args.Name+"_"+str(args.seed)
	print("model_name")
	print(model_name)
	accs=[]
	train_losses=[]
	test_losses=[]
	GFN_losses=[]
	actual_dropout_rates=[]
	for epoch in range(0, args.epochs + 1):
		start = time.time()


		test_loss,test_acc,masks=test(model, test_loader,args)	

		if args.Dropout in ["GFN", "Random"] and epoch in [0,5,10,20,50,100,150,200]:
		
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
			plt.savefig("images/"+model_name+'MasksSamples_'+str(epoch)+'_epoch.png')
			plt.clf()


		train_loss,GFN_loss,actual_dropout_rate=train(model, train_loader, optimizer, epoch,args)
		

		accs.append(test_acc)
		train_losses.append(train_loss)
		test_losses.append(test_loss)
		GFN_losses.append(GFN_loss)
		actual_dropout_rates.append(actual_dropout_rate)


		scheduler.step()


		end = time.time()
		
		print("epoch"+str(epoch)+" takes time:"+str(end - start))
		df = pd.DataFrame({'test_acc':accs,'test_loss':test_losses,
							'train_loss':train_losses,'GFNLoss':GFN_losses,
							"actual_dropout_rates":actual_dropout_rates})

		df.to_csv("../Results/"+model_name+"_performance.csv")

		
	if epoch%10==0:
		torch.save(model.state_dict(), "../checkpoints/"+model_name+"_cifar_model.pt")



#####code related to GFFN mask generating

class RandomMaskGenerator(nn.Module):
	def __init__(self, dropout_rate):
		super().__init__()
		self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	def forward(self, x):
		return torch.bernoulli((1. - self.dropout_rate) * torch.ones(x.shape[0],x.shape[1])).to(self.device)

	def log_prob(self, x, m):
		dist = (1. - self.dropout_rate) * torch.ones(x.shape[0],x.shape[1]).to(self.device)
		probs = dist * m + (1. - dist) * (1. - m)
		return torch.log(probs).sum(1)


class MLPMaskGenerator(nn.Module):
	def __init__(self, indim,outdim, dropout_rate, hidden=None, activation=nn.LeakyReLU):
		super().__init__()
		self.num_unit = torch.tensor(outdim).type(torch.float32)
		if dropout_rate!=None:
			self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)
		self.mlp = MLP(
			in_dim=indim,
			out_dim=outdim,
			hidden=hidden,
			activation=activation,
		)

	def _dist(self, x):
		x=x.view(x.shape[0],-1)
		x = self.mlp(x)
		if self.train:
			T=0.01
		else:
			T=1.0
		x = torch.sigmoid(T*x)
		#x=1.0/(1+torch.exp(-T*x))
		dist=x
		#dist = (1. - self.dropout_rate) * self.num_unit * x / (x.sum(1).unsqueeze(1) + 1e-6)
		#dist = dist.clamp(0, 1)
		return dist

	def forward(self, x):
		return torch.bernoulli(self._dist(x))

	def log_prob(self, x, m):
		dist = self._dist(x)
		probs = dist * m + (1. - dist) * (1. - m)
		return torch.log(probs).sum(1)



def construct_random_mask_generators(
		n_layers,
		dropout_rate,
):
	mask_generators = nn.ModuleList()
	for i in range(n_layers):
		mask_generators.append(
			RandomMaskGenerator(
				dropout_rate=dropout_rate,
			)
		)

	return mask_generators


def construct_mlp_mask_generators(
		input_shapes,
		dropout_rate,
		hidden=None,
		activation=nn.LeakyReLU
):
	mask_generators = nn.ModuleList()
	for input_shape_ in input_shapes:
		mask_generators.append(
			MLPMaskGenerator(
				indim=input_shape_[0]*input_shape_[1]*input_shape_[2],
				outdim=input_shape_[0],
				dropout_rate=dropout_rate,
				hidden=hidden,
				activation=activation
			)
		)

	return mask_generators



class MLP(nn.Module):
	def __init__(self, in_dim=784, out_dim=10, hidden=None, activation=nn.LeakyReLU):
		super().__init__()
		if hidden is None:
			hidden = [32, 32]
		h_old = in_dim
		self.fc = nn.ModuleList()
		for h in hidden:
			self.fc.append(nn.Linear(h_old, h))
			h_old = h
		self.out_layer = nn.Linear(h_old, out_dim)
		self.activation = activation

	def forward(self, x):
		for layer in self.fc:
			x = self.activation()(layer(x))
		x = self.out_layer(x)
		return x
if __name__ == "__main__":
	main()