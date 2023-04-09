import torch
import torch.nn as nn


import torch.nn.functional as F
from copy import deepcopy
import numpy as np

import torch.optim as optim
import torchbnn as bnn
import random

epsilon = 1e-7
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MLP_GFN(nn.Module):
	def __init__(self, input_dim=784, num_classes=10, N=60000, layer_dims=(1024,1024,1024),
				weight_decay=5e-4, lambas=(.1, .1, .1),
				 activation=nn.LeakyReLU,opt=None):
		super(MLP_GFN, self).__init__()

		self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.opt=opt
		self.layer_dims = layer_dims
		self.input_dim = input_dim
		self.N = N #will be used to multply the likihood term in the loss function in some methods
		self.num_classes=num_classes

		self.weight_decay = weight_decay#placeholder
		self.lambas = lambas#placeholder
		self.epoch = 0#placeholder
		self.elbo = torch.zeros(1)#placeholder
		
		#the chance of generating random mask and temperature can be used together 
		self.random_chance=0.0 #chances of using random mask during traning
		self.temperature=2 # temperature in sigmoid , high temperature more close the p to 0.5 for binary mask

		h_old = input_dim
		self.fc = nn.ModuleList()


		if opt.BNN==True:###if use Bayesian neural network backbone
			for i, dimh in enumerate(self.layer_dims):
				inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
				self.fc.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.01,
												in_features=inp_dim,
												 out_features=dimh))
			self.out_layer = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01,
				in_features=self.layer_dims[-1], out_features=num_classes)

		else:
			
			for i, dimh in enumerate(self.layer_dims):
				inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
				self.fc.append(nn.Linear(inp_dim, dimh))

			self.out_layer=nn.Linear(self.layer_dims[-1], num_classes)
	
 
		self.activation = activation




		###construct the mask for GFFN

		self.rand_mask_generator=RandomMaskGenerator(dropout_rate=opt.mlp_dr)


		self.p_zx_mask_generators=construct_conditional_mask_generators(layer_dims=layer_dims,
														additional_input_dims=[0 for j in layer_dims],
														hiddens=[32,32]).to(device)#p(z|x)

		self.q_zxy_mask_generators=construct_multiinput_conditional_mask_generators(layer_dims=layer_dims,
														additional_input_dims=[num_classes for j in layer_dims],
														hiddens=[32,32]).to(device)#q(z|x,y) 

	
		self.p_z_mask_generators=RandomMaskGenerator(dropout_rate=opt.mlp_dr)

		self.q_z_mask_generators=construct_unconditional_mask_generators(layer_dims=layer_dims,
														hiddens=[32,32]).to(device)#q(z)



		self.activation = activation

		mg_activation=nn.LeakyReLU
		

		###partition functions
		self.LogZ_total_flowestimator = MLP(in_dim=input_dim+num_classes,out_dim=1,hidden=[16],
									activation=mg_activation).to(device)#paritization function when the GFN condition on both x and y 


		self.LogZ_unconditional=nn.Parameter(torch.tensor(0.0))#paritization function when the GFN does not condition on any input 




		######optimizer
		z_lr=1e-1
		mg_lr_z=1e-3
		mg_lr_mu=1e-3
		lr=opt.lr
		self.beta=self.opt.beta #temperature on rewards


		
		
		q_z_param_list = [{'params': self.q_z_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
							{'params': self.LogZ_unconditional, 'lr': z_lr,"weight_decay":0.1}]
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


		taskmodel_param_list = [{'params': self.fc.parameters(), 'lr': lr},
		{'params': self.out_layer.parameters(), 'lr': lr}]

		self.taskmodel_optimizer = optim.Adam(taskmodel_param_list)
		self.taskmodel_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.taskmodel_optimizer,
		 milestones=self.opt.schedule_milestone,gamma=0.2)


		self.to(self.device)



	def score(self, x):
		return self.output(x.view(-1, self.input_dim))

	def forward(self, x, y,mask):


		#using GFlownet

		x=x.view(-1, self.input_dim)
		
		if self.training:
			logits,actual_masks,masks_qz,masks_conditional,LogZ_unconditional,LogPF_qz,LogR_qz,LogPB_qz,LogPF_BNN,LogZ_conditional,LogPF_qzxy,LogR_qzxy,LogPB_qzxy,Log_pzx,Log_pz =self.GFN_forward(x,y,mask)

		else:

			####the inference code all ready set certain number of repeats, so set to 1 here
			N_repeats=1#sample multiple times and use average as inference prediciton because GFN cannot take expection easily
			logits=[]
			for _ in range(N_repeats):
				logits_,actual_masks,masks_qz,masks_conditional,LogZ_unconditional,LogPF_qz,LogR_qz,LogPB_qz,LogPF_BNN,LogZ_conditional,LogPF_qzxy,LogR_qzxy,LogPB_qzxy,Log_pzx,Log_pz =self.GFN_forward(x,y,mask)
				logits.append(logits_.unsqueeze(2))
			logits=torch.logsumexp(torch.cat(logits,2),2) 
			
		return logits,actual_masks



	def regularization(self):
		#place holder
		regularization = 0.
		
		return regularization

	def prune_rate(self):
		#placeholder to fit in the pipeline
		return 0.0



	#####GFN related functions
	def GFN_forward(self, x,y,mask="none"):
		###during inference y are not used 

		x=x.reshape(x.shape[0],-1)
		y=torch.nn.functional.one_hot(y, self.num_classes)#convert to one hot vector
		batch_size,input_dim=x.shape[0],x.shape[1]
		
			
		LogZ_unconditional=self.LogZ_unconditional#top down mask has partition functionindepdent of input x
		LogZ_conditional=self.LogZ_total_flowestimator(torch.cat([x,y],1))#partition function for bottomp mask is input dependent
		
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

		if self.training:#use tempered version of the policy q(z) or q(z|x,y) during training, becareful of the difference between "train" and "training"
			temperature=self.temperature
		else:
			temperature=1.0
		masks_qz=[[] for _ in range(len(self.fc))]
		
		for layer_idx in range(len(self.fc)):

			if ("topdown" in mask):
				EPSILON=random.uniform(0,1)
				if layer_idx==0:
					#during ranom random action+ tempered policy is used
					if (EPSILON<self.random_chance) and (self.training):
						qz_mask_l=self.rand_mask_generator(torch.zeros(batch_size,input_dim).to(device),temperature).to(device)

					else:
						qz_mask_l=self.q_z_mask_generators[layer_idx](torch.zeros(batch_size,input_dim).to(device),temperature)      
					
					qz_p_l=self.q_z_mask_generators[layer_idx].prob(torch.zeros(batch_size,input_dim).to(device),qz_mask_l) 
				else:
					##concatenate all previous masks
					previous_mask=[]
					for j in range(layer_idx):
						previous_mask.append(masks_qz[j][-1])
					previous_mask=torch.cat(previous_mask,1)

					if (EPSILON<self.random_chance) and (self.training):
						qz_mask_l=self.rand_mask_generator(previous_mask).to(device)
					else:
						qz_mask_l=self.q_z_mask_generators[layer_idx](previous_mask,temperature)
					qz_p_l=self.q_z_mask_generators[layer_idx].prob(previous_mask,qz_mask_l)
				masks_qz[layer_idx].append(qz_mask_l.detach().clone())
				

				
				LogPF_qz+=(qz_mask_l*torch.log(qz_p_l)+(1-qz_mask_l)*torch.log(1-qz_p_l)).sum(1)

				LogPB_qz+=0 #uniform backward P
			else:
				masks_qz[layer_idx].append(torch.ones(self.layer_dims[layer_idx]).to(device))
				

		'''
		forward pass
		'''
		actual_masks=[]
		masks_conditional=[]
		for layer_idx in range(len(self.fc)):
			layer=self.fc[layer_idx]
		

			# if epsilon>0 and random.uniform(0, 1)<epsilon:
			#     m_mu_l=torch.bernoulli(torch.zeros(m_mu_l.shape).fill_(0.5).to(device))#generate random masks                
			

			####version sample weights and biases
			if self.opt.BNN:

				weight_ = self.fc[layer_idx].weight_mu+ torch.exp(self.fc[layer_idx].weight_log_sigma) * torch.randn_like(self.fc[layer_idx].weight_log_sigma).to(device) 
				bias_ = self.fc[layer_idx].bias_mu + torch.exp(self.fc[layer_idx].bias_log_sigma) * torch.randn_like(self.fc[layer_idx].bias_log_sigma).to(device)

				x=F.linear(x, weight_, bias_)#the same weights and biases are used for the WHOLE batch

				pred = self.activation()(x)

				####probabilitty of picking Bayesian neural network 
				log_pdf_weights=torch.distributions.normal.Normal(self.fc[layer_idx].weight_mu,torch.exp(self.fc[layer_idx].weight_log_sigma)).log_prob(weight_)  

	

				log_pdf_bias=torch.distributions.normal.Normal(self.fc[layer_idx].bias_mu,torch.exp(self.fc[layer_idx].bias_log_sigma)).log_prob(bias_)

				###add P_F_wb to the GFN loss
				LogPF_BNN+=(log_pdf_weights.mean()+log_pdf_bias.mean()).repeat(batch_size)


			else:
				x=layer(x)
				x = self.activation()(x)


			#####different masks generator
			if "bottomup" in mask:

				if self.training:
					#during training use q(z|x,y;phi) to sample mask
					#print("using q(z|x,y;phi)") #make sure the right function was used
					EPSILON=random.uniform(0,1)
					if layer_idx==0:

						if EPSILON>=self.random_chance:
							m_conditional_l= self.q_zxy_mask_generators[layer_idx](torch.zeros(batch_size,x.shape[1]).to(device),x.clone().detach(),y.float().clone().detach(),temperature)#generate mask based on activation from previous layer, detach from BNN training
						else:
							m=self.rand_mask_generator(torch.zeros(batch_size,x.shape[1]).to(device)).to(device)
							m_conditional_l=m
						qzxy_p_l= self.q_zxy_mask_generators[layer_idx].prob(torch.zeros(batch_size,x.shape[1]).to(device),x.clone().detach(),y.float().clone().detach(),m_conditional_l)
						
					else:
						previous_actual_mask=[]#use previous actual masks
						for j in range(layer_idx):
							previous_actual_mask.append(actual_masks[j])
						previous_actual_mask=torch.cat(previous_actual_mask,1)
						
						if EPSILON>=self.random_chance:
							m_conditional_l = self.q_zxy_mask_generators[layer_idx](previous_actual_mask,x.clone().detach(),y.float().clone().detach(),temperature)
						else:# during training ,of a centain chance a random policy will be used to explore the space
								
							m=self.rand_mask_generator(torch.zeros(batch_size,x.shape[1]).to(device)).to(device)
							m_conditional_l=m

						qzxy_p_l= self.q_zxy_mask_generators[layer_idx].prob(previous_actual_mask,x.clone().detach(),y.float().clone().detach(),m_conditional_l)

					masks_conditional.append(m_conditional_l)

					###add log P_F_Z to the GFN loss

					LogPF_qzxy+=(m_conditional_l*torch.log(qzxy_p_l)+(1-m_conditional_l)*torch.log(1-qzxy_p_l)).sum(1)

					LogPB_qzxy-=0
				

				else:
					#during inference use p(z|x;xi) to sample mask
						#print("p(z|x;xi)") #make sure the right function was used
						if layer_idx==0:
							m_conditional_l= self.p_zx_mask_generators[layer_idx](x.clone().detach())
						else:
							previous_actual_mask=[]#use previous actual masks
							for j in range(layer_idx):
								previous_actual_mask.append(actual_masks[j])

							###calculate p(z|x;xi)
							input_pzx=torch.cat(previous_actual_mask+[x.clone().detach()],1)

							m_conditional_l= self.p_zx_mask_generators[layer_idx](input_pzx)#generate mask based on activation from previous layer, detach from BNN training


						masks_conditional.append(m_conditional_l)

			else:
					masks_conditional.append(torch.ones(x.shape).to(device))

				



			

			if mask=="random":# completely random mask used
				EPSILON=random.uniform(0,1)
				m=self.rand_mask_generator(x).to(device)

			elif mask=="topdown":
				m_qz_l=masks_qz[layer_idx][-1]  
				m=m_qz_l
			elif mask=="bottomup":   
				m=m_conditional_l

			elif mask=="none":
				m=torch.ones(x.shape).to(device)

			m=m.detach().clone()#mask should not interfere backbone model(MLP or resnet etc) training
			#rescale the output
			if mask!="none":
				multipliers = m.shape[1] / (m.sum(1) + 1e-6)
				x = torch.mul((x * m).T, multipliers).T

			actual_masks.append(m)


			###calculate p(z;xi) and p(z|x;xi) both of which are needed for training
			if layer_idx==0:
				
				###calculate p(z|x;xi)
				Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(x.clone().detach(),m)
				#calculate p(z)
				Log_P_z_l = self.p_z_mask_generators.log_prob(m,m)
				
			else:
				previous_actual_mask=[]#use previous actual masks
				for j in range(layer_idx):
					previous_actual_mask.append(actual_masks[j])

				###calculate p(z|x;xi)
				input_pzx=torch.cat(previous_actual_mask+[x.clone().detach()],1)

				Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(input_pzx,m)#generate mask based on activation from previous layer, detach from BNN training


				###calculate p(zxi)
				input_pz=torch.cat(previous_actual_mask,1)
		
				Log_P_z_l = self.p_z_mask_generators.log_prob(m,m)

			Log_pzx+=Log_P_zx_l
			Log_pz+=Log_P_z_l



		####output final prediction

		if self.opt.BNN:
			###apply the task model
			weight_ = self.out_layer.weight_mu + torch.exp(self.out_layer.weight_log_sigma) * torch.randn_like(self.out_layer.weight_log_sigma).to(device) 
			bias_ = self.out_layer.bias_mu + torch.exp(self.out_layer.bias_log_sigma) * torch.randn_like(self.out_layer.bias_log_sigma).to(device)

			x=F.linear(x, weight_, bias_)

			logits = self.activation()(x)

			####probabilitty of picking Bayesian neural network 
			log_pdf_weights=torch.distributions.normal.Normal(self.out_layer.weight_mu,torch.exp(self.out_layer.weight_log_sigma)).log_prob(weight_)  

			# if layer_idx==1 and random.uniform(0, 1)<0.01:
			#     print(layer.weight_mu)

			log_pdf_bias=torch.distributions.normal.Normal(self.out_layer.bias_mu,torch.exp(self.out_layer.bias_log_sigma)).log_prob(bias_)

			###add P_F_wb to the GFN loss
			LogPF_BNN+=(log_pdf_weights.mean()+log_pdf_bias.mean()).repeat(batch_size)

		else:

			logits = self.out_layer(x)

	 

 
		return logits,actual_masks,masks_qz,masks_conditional,LogZ_unconditional,LogPF_qz,LogR_qz,LogPB_qz,LogPF_BNN,LogZ_conditional,LogPF_qzxy,LogR_qzxy,LogPB_qzxy,Log_pzx,Log_pz



	def _gfn_step(self, x, y,mask_train="",mask="none"):
		#####this step allows us to use different x,y to generate mask and calcualte reward(loss)

 
		metric = {}
		logits,actual_masks,masks_qz,masks_conditional,LogZ_unconditional,LogPF_qz,LogR_qz,LogPB_qz,LogPF_BNN,LogZ_conditional,LogPF_qzxy,LogR_qzxy,LogPB_qzxy,Log_pzx,Log_pz = self.GFN_forward(x,y,mask)
		
		#loss calculation
		#CEloss = F.nll_loss(logits, y)
 
		CEloss = nn.CrossEntropyLoss(reduction='none')(logits, y)
		LL=-CEloss
		
		print("self.beta",self.beta)


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



			self.q_z_optimizer.step()
			self.taskmodel_optimizer.step()
	

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
		for layer_idx in range(len(self.fc)):
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



#####code related to GFN mask generating



class RandomMaskGenerator(nn.Module):
	def __init__(self, dropout_rate):
		super().__init__()
		self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)

	def forward(self, x):
		return torch.bernoulli((1. - self.dropout_rate) * torch.ones(x.shape))

	def log_prob(self, x, m):
		dist = (1. - self.dropout_rate) * torch.ones(x.shape).to(device)
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

		
		probs_sampled=self._dist(x,T)#sample using tempered version of policy

		return torch.bernoulli(probs_sampled)

	def log_prob(self, x, m):
		dist = self._dist(x)
		probs = dist * m + (1. - dist) * (1. - m)
		return torch.log(probs).sum(1)

	def prob(self, x, m):
		dist = self._dist(x,1.0) #calculate prob using untempered policy
		probs = dist * m + (1. - dist) * (1. - m)
		return probs


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

		
		probs_sampled=self._dist(x1,x2,x3,T)#sample the mask using tempered policy

		return torch.bernoulli(probs_sampled)

	def log_prob(self, x1,x2,x3, m):
		dist = self._dist(x1,x2)
		probs = dist * m + (1. - dist) * (1. - m)
		return torch.log(probs).sum(1)

	def prob(self, x1,x2,x3, m):
		dist = self._dist(x1,x2,x3,1.0)#calculate probability using original policy always
		probs = dist * m + (1. - dist) * (1. - m)
		return probs


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
			in_dim=layer_dims[layer_idx]+additional_input_dims[layer_idx]
			out_dim=layer_dims[layer_idx]
		
		else:
			in_dim=additional_input_dims[layer_idx]
			for j in range(layer_idx):
				 in_dim+=layer_dims[j]
			in_dim+=layer_dims[layer_idx]

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


def construct_multiinput_conditional_mask_generators(
		layer_dims,
		additional_input_dims,
		hiddens=None,
		activation=nn.LeakyReLU
):
	mask_generators = nn.ModuleList()
	for layer_idx in range(len(layer_dims)):

		if layer_idx==0:
			in_dim1=layer_dims[layer_idx]
			in_dim2=layer_dims[layer_idx]
			in_dim3=additional_input_dims[layer_idx]
			
			out_dim=layer_dims[layer_idx]
		
		else:

			in_dim2=layer_dims[layer_idx]
			in_dim3=additional_input_dims[layer_idx]
			
			in_dim1=0

			for j in range(layer_idx):
				 in_dim1+=layer_dims[j]
			

			out_dim=layer_dims[layer_idx]

		mask_generators.append(
			multiMLPMaskGenerator(
				in_dim_1=in_dim1,in_dim_2=in_dim2,in_dim_3=in_dim3,
				out_dim=out_dim,
				hidden=hiddens,
				activation=activation
			)
		)

	return mask_generators

