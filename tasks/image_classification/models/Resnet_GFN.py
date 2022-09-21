
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

import random
import numpy as np
import torch.optim as optim

from torchvision import models, datasets, transforms

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ResNet_GFN(nn.Module):
	# droprate_init = 0.3
	def __init__(self, num_classes=10,lambas=0,weight_decay=0,activation=nn.LeakyReLU,opt=None):
		super(ResNet_GFN, self).__init__()
		
		self.opt=opt

		self.num_classes=num_classes
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		#BNN version need to be coded
		
		if opt.use_pretrained:
			self.resnet = models.resnet18(pretrained=True, num_classes=1000)#the pretrained version has 1000 classes
		
			self.resnet.fc = nn.Linear(512, num_classes)
		else:
			
			self.resnet = models.resnet18(pretrained=False, num_classes=num_classes)
		
		self.resnet.conv1 = torch.nn.Conv2d(
			3, 64, kernel_size=3, stride=1, padding=1, bias=False
		)
		self.resnet.maxpool = torch.nn.Identity()

		self.to(self.device)



		####GFN related
		#the chance of generating random mask and temperature can be used together 
		self.random_chance=0.1 #chances of using random mask during traning
		self.temperature=2 # temperature in sigmoid , high temperature more close the p to 0.5 for binary mask


		maskgenerator_input_shapes=[(512,4,4),(512,4,4)]#only apply dropout out on the last two blocks of resent18
		self.maskgenerator_input_shapes=maskgenerator_input_shapes
	   
	

		###construct the mask for GFFN

		self.rand_mask_generator=RandomMaskGenerator(dropout_rate=opt.mlp_dr)

		hiddens=[32,32]
		self.p_zx_mask_generators=construct_conditional_mask_generators(layer_dims=maskgenerator_input_shapes,
														additional_input_dims=[0 for j in maskgenerator_input_shapes],
														hiddens=hiddens).to(device)#p(z|x)

		self.q_zxy_mask_generators=construct_multiinput_conditional_mask_generators(
														n_channels=[DIMs[0] for DIMs in maskgenerator_input_shapes],
														layer_dims=[DIMs[0]*DIMs[1]*DIMs[2] for DIMs in maskgenerator_input_shapes],
														additional_input_dims=[num_classes for j in maskgenerator_input_shapes],
														hiddens=[32,32]).to(device)#q(z|x,y) 

	
		self.p_z_mask_generators=construct_unconditional_mask_generators(layer_dims=[x[0] for x in maskgenerator_input_shapes],
														hiddens=hiddens).to(device)#p(z)

		self.q_z_mask_generators=construct_unconditional_mask_generators(layer_dims=[x[0] for x in maskgenerator_input_shapes],
														hiddens=hiddens).to(device)#q(z)



		self.activation = activation

		mg_activation=nn.LeakyReLU
		

		###partition functions
		self.LogZ_total_flowestimator = CNN_MLP(CNN_in_dim=(3,32,32),mlp_in_dim=num_classes,out_dim=1,
									activation=mg_activation).to(device)#paritization function when the GFN condition on both x and y 


		self.LogZ_unconditional=nn.Parameter(torch.tensor(0.0))#paritization function when the GFN does not condition on any input 




		######optimizer
		z_lr=1e-1
		mg_lr_z=1e-3
		mg_lr_mu=1e-3
		lr=opt.lr
		self.beta=1 #temperature on rewards


		
		p_z_param_list = [{'params': self.p_z_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},]
		self.p_z_optimizer = optim.Adam(p_z_param_list)
	   
		q_z_param_list = [{'params': self.q_z_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
							{'params':self.LogZ_unconditional, 'lr': z_lr,"weight_decay":0.1}]
		self.q_z_optimizer = optim.Adam(q_z_param_list)


		p_zx_param_list = [{'params': self.p_zx_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},]
		self.p_zx_optimizer = optim.Adam(p_zx_param_list)
	   
		q_zxy_param_list = [{'params': self.q_zxy_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
							{'params': self.LogZ_total_flowestimator.parameters(), 'lr': z_lr,"weight_decay":0.1}]
		self.q_zxy_optimizer = optim.Adam(q_zxy_param_list)



		# GFN_Z_param_list = [{'params': self.LogZ_total_flowestimator, 'lr': z_lr,"weight_decay":0.1},
		# {'params': self.LogZ_unconditional.parameters(), 'lr': z_lr,"weight_decay":0.1}]

		# self.GFN_Z_optimizer = optim.Adam(GFN_Z_param_list)

		# self.GFN_Z_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.GFN_Z_optimizer, milestones=[200,300,350],gamma=0.2)

		if self.opt.Tune_last_layer_only:
			taskmodel_param_list = [{'params': self.resnet.fc.parameters(), 'lr': lr}]
		else:
			taskmodel_param_list = [{'params': self.resnet.parameters(), 'lr': lr}]

		self.taskmodel_optimizer = optim.Adam(taskmodel_param_list)
		self.taskmodel_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.taskmodel_optimizer,
		 milestones=self.opt.schedule_milestone,gamma=0.2)


		self.to(self.device)
	  


		##the following are place holder to be consistent with other codes

		self.n = 4 # 4
		self.N = 0
		self.beta_ema = 0
		self.epoch = 0
		self.elbo = torch.zeros(1)
		
		#droprate_init = self.opt.wrn_dr

		self.weight_decay = 0
		self.lamba = lambas

	def forward(self, x, y,mask):


		#using GFlownet

		
		if self.training:
			logits,actual_masks,masks_qz,masks_qzxy,LogZ_unconditional,LogPF_qz,LogR_qz,LogPB_qz,LogPF_BNN,LogZ_conditional,LogPF_qzxy,LogR_qzxy,LogPB_qzxy,Log_pzx,Log_pz =self.GFN_forward(x,y,mask)

		else:

			####the inference code all ready set certain number of repeats, so set to 1 here
			N_repeats=1#sample multiple times and use average as inference prediciton because GFN cannot take expection easily
			logits=[]
			for _ in range(N_repeats):
				logits_,actual_masks,masks_qz,masks_qzxy,LogZ_unconditional,LogPF_qz,LogR_qz,LogPB_qz,LogPF_BNN,LogZ_conditional,LogPF_qzxy,LogR_qzxy,LogPB_qzxy,Log_pzx,Log_pz =self.GFN_forward(x,y,mask)
				logits.append(logits_.unsqueeze(2))
			logits=torch.logsumexp(torch.cat(logits,2),2) 
			
		return logits


 

	#####GFN related functions
	def GFN_forward(self, x,y,mask="none"):
		###during inference y are not used 

		
		y=torch.nn.functional.one_hot(y, self.num_classes).float()#convert to one hot vector
		batch_size,input_dim=x.shape[0],x.shape[1]
		
			
		LogZ_unconditional=self.LogZ_unconditional#top down mask has partition functionindepdent of input x
	 
		LogZ_conditional=self.LogZ_total_flowestimator(x,y)#partition function for bottomp mask is input dependent
		
		LogPF_qz=torch.zeros(batch_size).to(device)# forward probability, unconditonal mask
		LogPB_qz=torch.zeros(batch_size).to(device)# backward prob
		LogR_qz=torch.zeros(batch_size).to(device)

		LogPF_BNN=torch.zeros(batch_size).to(device)
		LogPB_BNN=torch.zeros(batch_size).to(device)

		LogPF_qzxy=torch.zeros(batch_size).to(device)# forward probability, for contional mask generator
		LogR_qzxy=torch.zeros(batch_size).to(device)
		LogPB_qzxy=torch.zeros(batch_size).to(device)

		Log_pzx=torch.zeros(batch_size).to(device)#part of log R for bottom up mask
		Log_pz=torch.zeros(batch_size).to(device)#part of log R for topdown mask

		'''
		mu mask generation, indepedent of the input x 
		'''

		#initialize masks as all zeros(dropout them all)
		#one batch share the same mu mask

		if self.train:#use tempered version of the policy q(z) or q(z|x,y) during training
			temperature=self.temperature
		else:
			temperature=1.0

		
		masks_qz=[[] for _ in range(len(self.maskgenerator_input_shapes))]
		
		for layer_idx in range(len(self.maskgenerator_input_shapes)):
		
			if "topdown"==mask:
				if layer_idx==0:
					qz_mask_l,qz_p_l=self.q_z_mask_generators[layer_idx](torch.zeros(batch_size,784).to(device),temperature)#784 is an arbitary number here      
					
				else:
					##concatenate all previous masks
					previous_mask=[]
					for j in range(layer_idx):
						previous_mask.append(masks_qz[j][-1])
					previous_mask=torch.cat(previous_mask,1)
					qz_mask_l,qz_p_l=self.q_z_mask_generators[layer_idx](previous_mask,temperature)
				
				masks_qz[layer_idx].append(qz_mask_l.detach().clone())
				

				
				LogPF_qz+=(qz_mask_l*torch.log(qz_p_l)+(1-qz_mask_l)*torch.log(1-qz_p_l)).sum(1)

				LogPB_qz+=0 #uniform backward P
			else:
				masks_qz[layer_idx].append(torch.ones(self.maskgenerator_input_shapes[layer_idx][0]).to(device))
				

		'''
		forward pass
		'''
		actual_masks=[]
		masks_conditional=[]

		if self.opt.BNN:

			print("to be defined")

		else:
				
			x = self.resnet.conv1(x)
			x=self.resnet.bn1(x)
			x=self.resnet.relu(x)
			x=self.resnet.maxpool(x)

   
			##resnet18 has 4 layers, each with 2 blocks
			block_idx=0
			for layer in [self.resnet.layer1,self.resnet.layer2,self.resnet.layer3,self.resnet.layer4]:#number of layers changes over architecture

				for blockid in range(2):##number block per layer changes over architecture
					
					
					
					identity=x
					out=layer[blockid].conv1(x)
					out=layer[blockid].bn1(out)
					out=layer[blockid].relu(out)

					
					out=layer[blockid].conv2(out)
					out=layer[blockid].bn2(out)

					if layer[blockid].downsample is not None:

						identity = layer[blockid].downsample(x)




					#####different masks generator
					if block_idx>=6:
				
						layer_idx=block_idx-6
						if ("bottomup" in mask) or ("upNdown" in mask):

							if self.train:
								#during training use q(z|x,y;phi) to sample mask
								if layer_idx==0:

									m_conditional_l,qzxy_p_l= self.q_zxy_mask_generators[layer_idx](torch.zeros(batch_size,out.shape[1]).to(device),out.reshape(batch_size,-1).clone().detach(),y.float().clone().detach(),temperature)#generate mask based on activation from previous layer, detach from BNN training
									
									
								else:
									previous_actual_mask=[]#use previous actual masks
									for j in range(layer_idx):
										previous_actual_mask.append(actual_masks[j])
									previous_actual_mask=torch.cat(previous_actual_mask,1)
									
									m_conditional_l,qzxy_p_l = self.q_zxy_mask_generators[layer_idx](previous_actual_mask,out.reshape(batch_size,-1).clone().detach(),y.float().clone().detach(),temperature)
						

								masks_conditional.append(m_conditional_l)

								###add log P_F_Z to the GFN loss

								LogPF_qzxy+=(m_conditional_l*torch.log(qzxy_p_l)+(1-m_conditional_l)*torch.log(1-qzxy_p_l)).sum(1)

								LogPB_qzxy-=0
							

							else:
								#during inference use p(z|x;xi) to sample mask
									if layer_idx==0:
										m_conditional_l,_= Log_P_zx_l = self.p_zx_mask_generators[layer_idx](out.clone().detach())
									else:
										previous_actual_mask=[]#use previous actual masks
										for j in range(layer_idx):
											previous_actual_mask.append(actual_masks[j])

										###calculate p(z|x;xi)
										input_pzx=torch.cat(previous_actual_mask+[x.clone().detach()],1)

										m_conditional_l,_= self.p_zx_mask_generators[layer_idx](input_pzx)#generate mask based on activation from previous layer, detach from BNN training


									masks_conditional.append(m_conditional_l)

						else:
								masks_conditional.append(torch.ones(x.shape).to(device))

							



						EPSILON=random.uniform(0,1)

						if mask=="random" or (EPSILON<self.random_chance and self.training):# during training ,of a centain chance a random policy will be used to explore the space
							
							m=self.rand_mask_generator(torch.zeros(out.shape[0],out.shape[1])).to(device)

						elif mask=="topdown":
							m_qz_l=masks_qz[layer_idx][-1]  
							m=m_qz_l
						elif mask=="bottomup" or mask=="upNdown":   
							m=m_conditional_l


						elif mask=="none":
							m=torch.ones(out.shape[0],out.shape[1]).to(device)

						m=m.detach().clone()#mask should not interfere backbone model(MLP or resnet etc) training
						
						###calculate p(z;xi) and p(z|x;xi) both of which are needed for training
						if layer_idx==0:
							
							###calculate p(z|x;xi)
							Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(out.reshape(batch_size,-1).clone().detach(),m)
							#calculate p(z|xi)
							Log_P_z_l = self.p_z_mask_generators[layer_idx].log_prob(torch.zeros(batch_size,784).to(device),m)
							
						else:
							previous_actual_mask=[]#use previous actual masks
							for j in range(layer_idx):
								previous_actual_mask.append(actual_masks[j])

							###calculate p(z|x;xi)
							input_pzx=torch.cat(previous_actual_mask+[out.reshape(batch_size,-1).clone().detach()],1)

							Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(input_pzx,m)#generate mask based on activation from previous layer, detach from BNN training


							###calculate p(zxi)
							input_pz=torch.cat(previous_actual_mask,1)
					
							Log_P_z_l = self.p_z_mask_generators[layer_idx].log_prob(input_pz,m)#generate mask based on activation from previous layer, detach from BNN training


						Log_pzx+=Log_P_zx_l
						Log_pz+=Log_P_z_l

						###apply the mask
			 

						out=out.mul(m.unsqueeze(2).unsqueeze(3)) 


						actual_masks.append(m)

					out += identity
					out = layer[blockid].relu(out)
					x=out
					
					block_idx+=1






		####fc layer 

		x=self.resnet.avgpool(x)
		x = torch.flatten(x, 1)
		x=self.resnet.fc(x) 
		pred=x
		#pred = F.log_softmax(x, dim=1)   


 
		return pred,actual_masks,masks_qz,masks_conditional,LogZ_unconditional,LogPF_qz,LogR_qz,LogPB_qz,LogPF_BNN,LogZ_conditional,LogPF_qzxy,LogR_qzxy,LogPB_qzxy,Log_pzx,Log_pz


	def _gfn_step(self, x, y,mask_train="",mask="none"):
		#####this step allows us to use different x,y to generate mask and calcualte reward(loss)

 
		metric = {}
		logits,actual_masks,masks_qz,masks_conditional,LogZ_unconditional,LogPF_qz,LogR_qz,LogPB_qz,LogPF_BNN,LogZ_conditional,LogPF_qzxy,LogR_qzxy,LogPB_qzxy,Log_pzx,Log_pz = self.GFN_forward(x,y,mask)
		
		#loss calculation
		#CEloss = F.nll_loss(logits, y)
		#CEloss = F.nll_loss(reduction='none')(logits, y)
		CEloss = nn.CrossEntropyLoss(reduction='none')(logits, y)
		LL=-CEloss
		
		

		LogR_unconditional=self.beta*self.N*LL.detach().clone()+Log_pz.detach().clone()
		GFN_loss_unconditional=(LogZ_unconditional+LogPF_qz-LogR_unconditional-LogPB_qz)**2#+kl_weight*kl#jointly train the last layer BNN and the mask generator
		
		LogR_conditional=self.beta*LL.detach().clone()+Log_pzx.detach().clone()
		GFN_loss_conditional=(LogZ_conditional+LogPF_qzxy-LogR_conditional-LogPB_qzxy)**2#+kl_weight*kl#jointly train the last layer BNN and the mask generator
		
		#LogR_upNdown=self.beta*self.N*(LL.detach().clone()+Log_pzx.detach().clone())+Log_pz.detach().clone()
		LogR_upNdown=self.beta*(LL.detach().clone())+Log_pzx.detach().clone()+Log_pz.detach().clone()
		GFN_loss_upNdown=(LogZ_conditional+LogPF_qzxy-LogR_upNdown-LogPB_qzxy)**2#+kl_weight*kl#jointly train the last layer BNN and the mask generator
		

		# Update model

		acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
		
		metric['CELoss'] = CEloss.mean().item()#loss output from the model
	 

 
		metric['GFN_loss_unconditional'] = GFN_loss_unconditional.mean().item()#loss output from the model is per sample
		metric['GFN_loss_conditional'] = GFN_loss_conditional.mean().item()#loss output from the model is per sample
		
		metric['acc'] = acc

		COR_qz=np.corrcoef(LogPF_qz.cpu().detach().numpy(),LogR_unconditional.cpu().detach().numpy())[0,1]
		
		metric['COR_qz'] =COR_qz

		COR_qzxy=np.corrcoef(LogPF_qzxy.cpu().detach().numpy(),LogR_conditional.cpu().detach().numpy())[0,1]
		
		metric['COR_qzxy'] =COR_qzxy

		if mask=="topdown":

			#train  q(z) and logZ by GFN loss
			self.q_z_optimizer.zero_grad()

			GFN_loss_unconditional.mean().backward(retain_graph=True)


			#train task model by maximizing ELBO
			if self.opt.BNN==True:

				#a different equation if using BNN
				
				kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

				kl_1 = kl_loss(self.fc)

				kl_2 = kl_loss(self.out_layer)

				kl=kl_1+kl_2

				metric['kl_loss']=(kl).item()

				taskmodel_loss=kl+self.N*CEloss+(LogPF_qzxy-Log_pzx)#loss BNN

				self.taskmodel_optimizer.zero_grad()

				taskmodel_loss.mean().backward(retain_graph=True)
				
				#self.taskmodel_optimizer.step()

			else:
				self.taskmodel_optimizer.zero_grad()

				taskmodel_loss=CEloss
				taskmodel_loss.mean().backward(retain_graph=True)

				#self.taskmodel_optimizer.step()


			##train p(z) by maximize EBLO

			self.p_z_optimizer.zero_grad()

			pz_loss=-Log_pz
			pz_loss.mean().backward(retain_graph=True)


			self.q_z_optimizer.step()
			self.taskmodel_optimizer.step()
			self.p_z_optimizer.step()

		if mask=="bottomup":

				#train  q(z|x,y) and logZ by GFN loss
				self.q_zxy_optimizer.zero_grad()

				GFN_loss_conditional.mean().backward(retain_graph=True)

				
				#train task model by maximizing ELBO
				if self.opt.BNN==True:

					#a different equation if using BNN
					
					kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

					kl_1 = kl_loss(self.fc)

					kl_2 = kl_loss(self.out_layer)

					kl=kl_1+kl_2

					metric['kl_loss']=(kl).item()

					taskmodel_loss=kl+self.N*CEloss+self.N*(LogPF_qzxy-Log_pzx)#loss BNN, in real practice detach BNN and qzxy,pzx trainng

					self.taskmodel_optimizer.zero_grad()

					taskmodel_loss.mean().backward(retain_graph=True)
					
					#self.taskmodel_optimizer.step()

				else:

					self.taskmodel_optimizer.zero_grad()

					taskmodel_loss=CEloss
					taskmodel_loss.mean().backward(retain_graph=True)

					#self.taskmodel_optimizer.step()


				##train p(z|x) by maximize EBLO

				self.p_zx_optimizer.zero_grad()

				pzx_loss=-Log_pzx
				pzx_loss.mean().backward(retain_graph=True)
				

				self.taskmodel_optimizer.step()

				self.q_zxy_optimizer.step()

				self.p_zx_optimizer.step()

		if mask=="upNdown":

				#train  q(z|x,y) and logZ by GFN loss
				self.q_zxy_optimizer.zero_grad()

				GFN_loss_upNdown.mean().backward(retain_graph=True)

				
				#train task model by maximizing ELBO
				if self.opt.BNN==True:

					#a different equation if using BNN
					
					kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

					kl_1 = kl_loss(self.fc)

					kl_2 = kl_loss(self.out_layer)

					kl=kl_1+kl_2

					metric['kl_loss']=(kl).item()

					taskmodel_loss=kl+self.N*CEloss+self.N*(LogPF_qzxy-Log_pzx)#loss BNN, in real practice detach BNN and qzxy,pzx trainng

					self.taskmodel_optimizer.zero_grad()

					taskmodel_loss.mean().backward(retain_graph=True)
					
					#self.taskmodel_optimizer.step()

				else:

					self.taskmodel_optimizer.zero_grad()

					taskmodel_loss=CEloss
					taskmodel_loss.mean().backward(retain_graph=True)

					#self.taskmodel_optimizer.step()


				##train p(z|x) by maximize EBLO

				self.p_zx_optimizer.zero_grad()

				pzx_loss=-Log_pzx
				pzx_loss.mean().backward(retain_graph=True)



				###train p(z)
				self.p_z_optimizer.zero_grad()

				pz_loss=-Log_pz
				pz_loss.mean().backward(retain_graph=True)		



				self.taskmodel_optimizer.step()

				self.q_zxy_optimizer.step()

				self.p_zx_optimizer.step()

				self.p_z_optimizer.step()

		if mask=="random" or mask=="none":

				self.taskmodel_optimizer.zero_grad()

				taskmodel_loss=CEloss
				taskmodel_loss.mean().backward()

				self.taskmodel_optimizer.step()


		###calculated actual droppout rate
		actual_dropout_rate=0

		n_units=0
		n_dropped=0.0
		batch_size=x.shape[0]
		for layer_idx in range(len(self.maskgenerator_input_shapes)):
			m=actual_masks[layer_idx]
			n_units+=m.shape[1]
			
			n_dropped+=(m==0).float().mean(0).sum()

		actual_dropout_rate=n_dropped/n_units 
		metric['actual_dropout_rate']=actual_dropout_rate.item()


		#differnet terms of TB
		metric['LogZ_unconditional']=LogZ_unconditional.mean().item()
		
		metric['LogPF_qz']=LogPF_qz.mean().item()

		metric['LogR_qz']=LogR_qz.mean().item()
		
		metric['LogPB_qz']=LogPB_qz.mean().item()

		metric['Log_pz']=Log_pz.mean().item()


		metric['LogZ_conditional']=LogZ_conditional.mean().item()
		
		metric['LogPF_qzxy']=LogPF_qzxy.mean().item()

		metric['LogR_qzxy']=LogR_qzxy.mean().item()
		
		metric['LogPB_qzxy']=LogPB_qzxy.mean().item()

		metric['Log_pzx']=Log_pzx.mean().item()

		metric['LogPF_BNN']=LogPF_BNN.mean().item()
		

		return metric



#####code related to GFFN mask generating

class RandomMaskGenerator(nn.Module):
	def __init__(self, dropout_rate):
		super().__init__()
		self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	def forward(self, x):
		return torch.bernoulli((1. - self.dropout_rate) * torch.ones(x.shape))

	def log_prob(self, x, m):
		dist = (1. - self.dropout_rate) * torch.ones(x.shape).to(self.device)
		probs = dist * m + (1. - dist) * (1. - m)
		return torch.log(probs).sum(1)


class MLP(nn.Module):
	def __init__(self, in_dim=784, out_dim=10, hidden=None, activation=nn.LeakyReLU):
		super().__init__()
		if hidden is None:
			hidden = [32, 32]
		h_old = in_dim
		self.fc = nn.ModuleList()
  
		self.LN = nn.ModuleList()
		for h in hidden:
			self.fc.append(nn.Linear(h_old, h))
			self.LN.append(torch.nn.LayerNorm(h))
			h_old = h

		###layerNorm to cope with variation if inputs in


		self.out_layer = nn.Linear(h_old, out_dim)
		self.activation = activation

	def forward(self, x):

		for layer,ln in zip(self.fc,self.LN):
			x = self.activation()(layer(x))
			x=ln(x)
			
		x = self.out_layer(x)
		return x

class MLPMaskGenerator(nn.Module):
	def __init__(self, in_dim,out_dim, hidden=[32], activation=nn.LeakyReLU):
		super().__init__()
	   

		self.mlp = MLP(
			in_dim=in_dim,
			out_dim=out_dim,
			hidden=hidden,
			activation=activation,
		)

	def _dist(self, x,T=1.0):

		x = self.mlp(x)
	
		#x = torch.exp(x)/(1.0+torch.exp(x))
		x=nn.Sigmoid()(x/T)
	
		dist=x
		
		#dist = dist.clamp(1e-3, 1.0-1e-3)

		return dist

	def forward(self, x,T=1.0):

		
		probs=self._dist(x,T)

		return torch.bernoulli(probs), probs

	def log_prob(self, x, m):
		dist = self._dist(x)
		probs = dist * m + (1. - dist) * (1. - m)
		return torch.log(probs).sum(1)




class CNNMaskGenerator(nn.Module):
	def __init__(self, input_shape,hidden, activation=nn.LeakyReLU):
		super().__init__()
		self.num_unit = torch.tensor(input_shape[0]).type(torch.float32)

		self.CNN= CNN_(
			image_shape=input_shape,
			out_dim=input_shape[0],
			activation=activation,
			hidden=hidden,
		)

	def _dist(self, x):
		x = self.CNN(x)
 
		dist = torch.sigmoid(x)
		# dist = (1. - self.dropout_rate) * self.num_unit * x / (x.sum(1).unsqueeze(1) + 1e-6)
		# dist = dist.clamp(0, 1)
		return dist

	def forward(self, x):
		dist=self._dist(x)
	 
		return torch.bernoulli(self._dist(x)),dist

	def log_prob(self, x, m):
		dist = self._dist(x)
		probs = dist * m + (1. - dist) * (1. - m)
		return torch.log(probs).sum(1)



def construct_random_mask_generators(
		n_layers,
		dropout_rate,
):


	mask_generators = nn.ModuleList()
	for layer in range(n_layers):
		mask_generators.append(
			RandomMaskGenerator(
				dropout_rate=dropout_rate,
			)
		)

	return mask_generators


class CNN_(nn.Module):
	def __init__(self, image_shape=(160,32,32), out_dim=10, hidden=16*5*5, activation=nn.LeakyReLU):
		super().__init__()
   

		self.LN=nn.LayerNorm(image_shape)##normalize the input to prevent exploding
		
		n_in_channels=image_shape[0]
		self.conv1 = nn.Conv2d(n_in_channels, 6, 3)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 2)
		self.fc1 = nn.Linear(hidden, 32)
		self.fc2 = nn.Linear(32, 16)
		self.fc3 = nn.Linear(16, out_dim)
		self.activation = activation

		self.LN1=nn.LayerNorm(hidden)
		self.LN2=nn.LayerNorm(32)
		self.LN3=nn.LayerNorm(16)



	def forward(self, x):
		x=self.LN(x)
		x = self.pool(self.activation()(self.conv1(x)))
		
		x = self.pool(self.activation()(self.conv2(x)))

		x = torch.flatten(x, 1) # flatten all dimensions except batch
  
		x=self.LN1(x)
		x = self.activation()(self.fc1(x))
		x=self.LN2(x)
		x = self.activation()(self.fc2(x))
		x=self.LN3(x)
		x = self.fc3(x)

		return x

class multiMLPMaskGenerator(nn.Module):
	def __init__(self, in_dim_1,in_dim_2,in_dim_3,out_dim, hidden=[32], activation=nn.LeakyReLU):
		super().__init__()
	   
		####use two separete MLP to ensure y is not ignored
		self.mlp_1 = MLP(
			in_dim=in_dim_1,
			out_dim=10,
			hidden=hidden,
			activation=activation,
		)



		self.mlp_2 = MLP(
		in_dim=in_dim_2,
		out_dim=10,
		hidden=hidden,
		activation=activation,
		)

		self.mlp_3 = MLP(
			in_dim=in_dim_3,
			out_dim=10,
			hidden=hidden,
			activation=activation,
		)

		self.mlp_combine = MLP(
		in_dim=30,
		out_dim=out_dim,
		hidden=hidden,
		activation=activation,
		)

	def _dist(self, x1,x2,x3,T=1.0):

		x1 = self.mlp_1(x1)
		x2 = self.mlp_2(x2)
		x3 = self.mlp_3(x3)


		x=self.mlp_combine(torch.cat([x1,x2,x3],1))
	
		#x = torch.exp(x)/(1.0+torch.exp(x))
		x=nn.Sigmoid()(x/T)
	
		dist=x
		
		#dist = dist.clamp(1e-3, 1.0-1e-3)

		return dist

	def forward(self, x1,x2,x3,T=1.0):

		
		probs=self._dist(x1,x2,x3,T)

		return torch.bernoulli(probs), probs

	def log_prob(self, x1,x2,x3, m):
		dist = self._dist(x1,x2)
		probs = dist * m + (1. - dist) * (1. - m)
		return torch.log(probs).sum(1)


def construct_unconditional_mask_generators(
		layer_dims,
		hiddens=None,
		activation=nn.LeakyReLU
):
	mask_generators = nn.ModuleList()

	for layer_idx in range(len(layer_dims)):

		if layer_idx==0:
			in_dim=784
			out_dim=layer_dims[layer_idx]
		
		else:
			in_dim=0
			for j in range(layer_idx):
				 in_dim+=layer_dims[j]

			out_dim=layer_dims[layer_idx]

		
		mask_generators.append(
			MLPMaskGenerator(
				in_dim=in_dim,
				out_dim=out_dim,
				hidden=hiddens,
				activation=activation
			)
		)

	return mask_generators




def construct_conditional_mask_generators(
		layer_dims,
		additional_input_dims,
		hiddens=None,
		activation=nn.LeakyReLU
):
	mask_generators = nn.ModuleList()
	for layer_idx in range(len(layer_dims)):

		if layer_idx==0:
			in_dim=layer_dims[layer_idx][0]*layer_dims[layer_idx][1]*layer_dims[layer_idx][2]+additional_input_dims[layer_idx]
			out_dim=layer_dims[layer_idx][0]
		
		else:
			in_dim=additional_input_dims[layer_idx]
			for j in range(layer_idx):
				 in_dim+=layer_dims[j][0]
			in_dim+=layer_dims[layer_idx][0]*layer_dims[layer_idx][1]*layer_dims[layer_idx][2]

			out_dim=layer_dims[layer_idx][0]

		mask_generators.append(
			MLPMaskGenerator(
				in_dim=in_dim,
				out_dim=out_dim,
				hidden=hiddens,
				activation=activation
			)
		)

	return mask_generators


def construct_multiinput_conditional_mask_generators(
		n_channels,layer_dims,
		additional_input_dims,
		hiddens=None,
		activation=nn.LeakyReLU
):
	mask_generators = nn.ModuleList()
	for layer_idx in range(len(layer_dims)):

		if layer_idx==0:
			in_dim1=n_channels[layer_idx]
			in_dim2=layer_dims[layer_idx]
			in_dim3=additional_input_dims[layer_idx]
			
			out_dim=n_channels[layer_idx]
		
		else:

			in_dim2=layer_dims[layer_idx]
			in_dim3=additional_input_dims[layer_idx]
			
			in_dim1=0

			for j in range(layer_idx):
				 in_dim1+=n_channels[j]
			

			out_dim=n_channels[layer_idx]

		mask_generators.append(
			multiMLPMaskGenerator(
				in_dim_1=in_dim1,in_dim_2=in_dim2,in_dim_3=in_dim3,
				out_dim=out_dim,
				hidden=hiddens,
				activation=activation
			)
		)

	return mask_generators






class CNN_MLP(nn.Module):
	def __init__(self, CNN_in_dim,mlp_in_dim, out_dim=10, activation=nn.LeakyReLU):
		super().__init__()
   

		self.CNN=CNN_(image_shape=CNN_in_dim, out_dim=10,hidden=784)
		self.MLP=MLP(in_dim=mlp_in_dim, out_dim=10)
		self.MLP_combine=MLP(in_dim=20, out_dim=out_dim)


	def forward(self, x,y):
		
		'''
		x should have shape batch x channels x H x W 
		y should have shape batch x dim
		'''
		vec1=self.CNN(x)
		vec2=self.MLP(y)

		output=self.MLP_combine(torch.cat([vec1,vec2],1))

		return output





