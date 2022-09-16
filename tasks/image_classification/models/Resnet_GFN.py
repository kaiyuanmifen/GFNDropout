#from models.layer import ArmConv2d
#from models.layer.MAPConv2D import MAPConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
#from models.layer.MAPDense import MAPDense


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

        self.resnet = models.resnet18(pretrained=False, num_classes=10)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

        self.to(self.device)



        ####GFN related

        maskgenerator_input_shapes=[(512,4,4),(512,4,4)]#only apply dropout out on the last two blocks of resent18
        self.maskgenerator_input_shapes=maskgenerator_input_shapes
       
    

        ###construct the mask for GFFN

        self.rand_mask_generator=RandomMaskGenerator(dropout_rate=opt.mlp_dr)

        hiddens=[32,32]
        self.p_zx_mask_generators=construct_conditional_mask_generators(layer_dims=maskgenerator_input_shapes,
                                                        additional_input_dims=[0 for j in maskgenerator_input_shapes],
                                                        hiddens=hiddens).to(device)#p(z|x)

        self.q_zxy_mask_generators=construct_conditional_mask_generators(layer_dims=maskgenerator_input_shapes,
                                                        additional_input_dims=[num_classes for j in maskgenerator_input_shapes],
                                                        hiddens=hiddens).to(device)#q(z|x,y) 

    
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
                            {'params': self.LogZ_total_flowestimator.parameters(), 'lr': z_lr,"weight_decay":0.1}]
        self.q_z_optimizer = optim.Adam(q_z_param_list)


        p_zx_param_list = [{'params': self.p_zx_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},]
        self.p_zx_optimizer = optim.Adam(p_zx_param_list)
       
        q_zxy_param_list = [{'params': self.q_zxy_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
                            {'params': self.LogZ_unconditional, 'lr': z_lr,"weight_decay":0.1}]
        self.q_zxy_optimizer = optim.Adam(q_zxy_param_list)



        # GFN_Z_param_list = [{'params': self.LogZ_total_flowestimator, 'lr': z_lr,"weight_decay":0.1},
        # {'params': self.LogZ_unconditional.parameters(), 'lr': z_lr,"weight_decay":0.1}]

        # self.GFN_Z_optimizer = optim.Adam(GFN_Z_param_list)

        # self.GFN_Z_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.GFN_Z_optimizer, milestones=[200,300,350],gamma=0.2)


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

        
        masks_qz=[[] for _ in range(len(self.maskgenerator_input_shapes))]
        
        for layer_idx in range(len(self.maskgenerator_input_shapes)):

            if "topdown" in mask:
                if layer_idx==0:
                    qz_mask_l,qz_p_l=self.q_z_mask_generators[layer_idx](torch.zeros(batch_size,784).to(device))      
                    
                else:
                    ##concatenate all previous masks
                    previous_mask=[]
                    for j in range(layer_idx):
                        previous_mask.append(masks_qz[j][-1])
                    previous_mask=torch.cat(previous_mask,1)
                    qz_mask_l,qz_p_l=self.q_z_mask_generators[layer_idx](previous_mask)
                
                masks_qz[layer_idx].append(qz_mask_l.detach().clone())
                

                
                LogPF_qz+=(qz_mask_l*torch.log(qz_p_l)+(1-qz_mask_l)*torch.log(1-qz_p_l)).sum(1)

                LogPB_qz+=0 #uniform backward P
            else:
                masks_qz[layer_idx].append(torch.ones(self.maskgenerator_input_shapes[layer_idx][0]).to(device))
                

        '''
        forward pass
        '''
        actual_masks=[]
        masks_qzxy=[]

        if self.opt.BNN:

            print("to be defined")

        else:
                
            x = self.resnet.conv1(x)
            x=self.resnet.bn1(x)
            x=self.resnet.relu(x)
            x=self.resnet.maxpool(x)

            ### layer 1 
            x=self.resnet.layer1(x)
            

            ##layer 2 
  
            x=self.resnet.layer2(x)
            


            ##layer 3 
            x=self.resnet.layer3(x)
            

            ##layer 4 block 1

            identity=x
            out=self.resnet.layer4[0].conv1(x)
            out=self.resnet.layer4[0].bn1(out)
            out=self.resnet.layer4[0].relu(out)

            
            out=self.resnet.layer4[0].conv2(out)
            out=self.resnet.layer4[0].bn2(out)

            if self.resnet.layer4[0].downsample is not None:

                identity = self.resnet.layer4[0].downsample(x)




            #####generative mask
            layer_idx=0
            if "bottomup" in mask:
                if layer_idx==0:
                    m_qzxy_l,qzxy_p_l= self.q_zxy_mask_generators[layer_idx](torch.cat([out.reshape(batch_size,-1).clone().detach(),y.clone().detach()],1))#generate mask based on activation from previous layer, detach from BNN training
                    
                
                else:
                    previous_actual_mask=[]#use previous actual masks
                    for j in range(layer_idx):
                        previous_actual_mask.append(actual_masks[j])
                    input_qzxy=previous_actual_mask
                    input_qzxy.append(out.reshape(batch_size,-1).clone().detach())
                    input_qzxy.append(y.clone().detach())
                    input_qzxy=torch.cat(input_qzxy,1)
                    m_qzxy_l,qzxy_p_l = self.q_zxy_mask_generators[layer_idx](input_qzxy)
        

                masks_qzxy.append(m_qzxy_l)


                LogPF_qzxy+=(m_qzxy_l*torch.log(qzxy_p_l)+(1-m_qzxy_l)*torch.log(1-qzxy_p_l)).sum(1)

                LogPB_qzxy-=0
            else:
                masks_qzxy.append(torch.ones(x.shape).to(device))



            if mask=="topdown":
                m_qz_l=masks_qz[layer_idx][-1]  
                m=m_qz_l
            elif mask=="bottomup":   
                m=m_qzxy_l


            elif mask=="both":
                m_qz_l=masks_qz[layer_idx][-1]  
                m=m_qz_l*m_qzxy_l

            elif mask=="random":
                m=self.rand_mask_generator(torch.zeros(out.shape[0],out.shape[1]).to(device)).to(device)

            elif mask=="none":
                m=torch.ones(x.shape).to(device)

            m=m.detach().clone()#mask should not interfere backbone model(MLP or resnet etc) training
            ###apply the mask
 

            out=out.mul(m.unsqueeze(2).unsqueeze(3)) 


            actual_masks.append(m)
            out += identity
            out = self.resnet.layer4[0].relu(out)
            x=out
    

            ###calculate p(z;xi) and p(z|x;xi) both of which are needed for training
            if layer_idx==0:
                
                ###calculate p(z|x;xi)
                Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(x.reshape(batch_size,-1).clone().detach(),m)
                #calculate p(z|xi)
                Log_P_z_l = self.p_z_mask_generators[layer_idx].log_prob(torch.zeros(batch_size,784).to(device),m)
                
            else:
                previous_actual_mask=[]#use previous actual masks
                for j in range(layer_idx):
                    previous_actual_mask.append(actual_masks[j])

                ###calculate p(z|x;xi)
                input_pzx=torch.cat(previous_actual_mask+[x.reshape(batch_size,-1).clone().detach()],1)

                Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(input_pzx,m)#generate mask based on activation from previous layer, detach from BNN training


                ###calculate p(zxi)
                input_pz=torch.cat(previous_actual_mask,1)
        
                Log_P_z_l = self.p_z_mask_generators[layer_idx].log_prob(input_pz,m)#generate mask based on activation from previous layer, detach from BNN training


            Log_pzx+=Log_P_zx_l
            Log_pz+=Log_P_z_l




            ##layer 4 block 2

            identity=x
            out=self.resnet.layer4[1].conv1(x)
            out=self.resnet.layer4[1].bn1(out)
            out=self.resnet.layer4[1].relu(out)


            
            out=self.resnet.layer4[1].conv2(out)
            out=self.resnet.layer4[1].bn2(out)

            if self.resnet.layer4[1].downsample is not None:
                identity = resnet.layer4[1].downsample(x)




            #####generative mask
            layer_idx=1
            if "bottomup" in mask:
                if layer_idx==0:
                    m_qzxy_l,qzxy_p_l= self.q_zxy_mask_generators[layer_idx](torch.cat([out.reshape(batch_size,-1).clone().detach(),y.clone().detach()],1))#generate mask based on activation from previous layer, detach from BNN training
                    
                
                else:
                    previous_actual_mask=[]#use previous actual masks
                    for j in range(layer_idx):
                        previous_actual_mask.append(actual_masks[j])
                    input_qzxy=previous_actual_mask
                    input_qzxy.append(out.reshape(batch_size,-1).clone().detach())
                    input_qzxy.append(y.clone().detach())
                    input_qzxy=torch.cat(input_qzxy,1)
                    m_qzxy_l,qzxy_p_l = self.q_zxy_mask_generators[layer_idx](input_qzxy)
        

                masks_qzxy.append(m_qzxy_l)


                LogPF_qzxy+=(m_qzxy_l*torch.log(qzxy_p_l)+(1-m_qzxy_l)*torch.log(1-qzxy_p_l)).sum(1)

                LogPB_qzxy-=0
            else:
                masks_qzxy.append(torch.ones(x.shape).to(device))



            if mask=="topdown":
                m_qz_l=masks_qz[layer_idx][-1]  
                m=m_qz_l
            elif mask=="bottomup":   
                m=m_qzxy_l


            elif mask=="both":
                m_qz_l=masks_qz[layer_idx][-1]  
                m=m_qz_l*m_qzxy_l

            elif mask=="random":
                m=self.rand_mask_generator(torch.zeros(x.shape[0],x.shape[1]).to(device)).to(device)

            elif mask=="none":
                m=torch.ones(x.shape).to(device)

            m=m.detach().clone()#mask should not interfere backbone model(MLP or resnet etc) training
            ###apply the mask
            out=out.mul(m.unsqueeze(2).unsqueeze(3)) 

            actual_masks.append(m)
            out += identity
            out = self.resnet.layer4[1].relu(out)
            x=out
     

            ###calculate p(z;xi) and p(z|x;xi) both of which are needed for training
            if layer_idx==0:
                
                ###calculate p(z|x;xi)
                Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(x.reshape(batch_size,-1).clone().detach(),m)
                #calculate p(z|xi)
                Log_P_z_l = self.p_z_mask_generators[layer_idx].log_prob(torch.zeros(batch_size,784).to(device),m)
                
            else:
                previous_actual_mask=[]#use previous actual masks
                for j in range(layer_idx):
                    previous_actual_mask.append(actual_masks[j])

                ###calculate p(z|x;xi)
                input_pzx=torch.cat(previous_actual_mask+[x.reshape(batch_size,-1).clone().detach()],1)

                Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(input_pzx,m)#generate mask based on activation from previous layer, detach from BNN training


                ###calculate p(zxi)
                input_pz=torch.cat(previous_actual_mask,1)
        
                Log_P_z_l = self.p_z_mask_generators[layer_idx].log_prob(input_pz,m)#generate mask based on activation from previous layer, detach from BNN training


            Log_pzx+=Log_P_zx_l
            Log_pz+=Log_P_z_l


 

        ####fc layer 

        x=self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x=self.resnet.fc(x) 
        pred=x
        #pred = F.log_softmax(x, dim=1)   


 
        return pred,actual_masks,masks_qz,masks_qzxy,LogZ_unconditional,LogPF_qz,LogR_qz,LogPB_qz,LogPF_BNN,LogZ_conditional,LogPF_qzxy,LogR_qzxy,LogPB_qzxy,Log_pzx,Log_pz


    def _gfn_step(self, x, y,mask_train="",mask="none"):
        #####this step allows us to use different x,y to generate mask and calcualte reward(loss)

 
        metric = {}
        logits,actual_masks,masks_qz,masks_qzxy,LogZ_unconditional,LogPF_qz,LogR_qz,LogPB_qz,LogPF_BNN,LogZ_conditional,LogPF_qzxy,LogR_qzxy,LogPB_qzxy,Log_pzx,Log_pz = self.GFN_forward(x,y,mask)
        
        #loss calculation
        #CEloss = F.nll_loss(logits, y)
        #CEloss = F.nll_loss(reduction='none')(logits, y)
        CEloss = nn.CrossEntropyLoss(reduction='none')(logits, y)
        LL=-CEloss
        
        


        GFN_loss_unconditional=(LogZ_unconditional+LogPF_qz-self.beta*self.N*LL.detach().clone()-Log_pz.detach().clone()-LogPB_qz)**2#+kl_weight*kl#jointly train the last layer BNN and the mask generator
        
        GFN_loss_conditional=(LogZ_conditional+LogPF_qzxy-self.beta*LL.detach().clone()-Log_pzx.detach().clone()-LogPB_qzxy)**2#+kl_weight*kl#jointly train the last layer BNN and the mask generator
        
        # Update model

        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
        
        metric['CELoss'] = CEloss.mean().item()#loss output from the model
     

 
        metric['GFN_loss_unconditional'] = GFN_loss_unconditional.mean().item()#loss output from the model is per sample
        metric['GFN_loss_conditional'] = GFN_loss_conditional.mean().item()#loss output from the model is per sample
        
        metric['acc'] = acc


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

                taskmodel_loss=self.N*CEloss
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

                    taskmodel_loss=self.N*CEloss
                    taskmodel_loss.mean().backward(retain_graph=True)

                    #self.taskmodel_optimizer.step()


                ##train p(z|x) by maximize EBLO

                self.p_zx_optimizer.zero_grad()

                pzx_loss=-Log_pzx
                pzx_loss.mean().backward(retain_graph=True)
                

                self.taskmodel_optimizer.step()

                self.q_zxy_optimizer.step()

                self.p_zx_optimizer.step()



        if mask=="random":

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
        metric['LogZ_unconditional']=LogZ_conditional.mean().item()
        
        metric['LogPF_qz']=LogPF_qz.mean().item()

        metric['LogR_qz']=LogR_qz.mean().item()
        
        metric['LogPB_qz']=LogPB_qz.mean().item()


        metric['LogZ_conditional']=LogZ_conditional.mean().item()
        
        metric['LogPF_qzxy']=LogPF_qzxy.mean().item()

        metric['LogR_qzxy']=LogR_qzxy.mean().item()
        
        metric['LogPB_qzxy']=LogPB_qzxy.mean().item()

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



class MLPMaskGenerator(nn.Module):
    def __init__(self, in_dim,out_dim, hidden=[32], activation=nn.LeakyReLU):
        super().__init__()
       

        self.mlp = MLP_(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            activation=activation,
        )

    def _dist(self, x):

        x = self.mlp(x)
    
        #x = torch.exp(x)/(1.0+torch.exp(x))
        x=nn.Sigmoid()(x)
    
        dist=x
        
        #dist = dist.clamp(1e-3, 1.0-1e-3)

        return dist

    def forward(self, x):

             
        probs=self._dist(x)
 

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


class MLP_(nn.Module):
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




class CNN_MLP(nn.Module):
    def __init__(self, CNN_in_dim,mlp_in_dim, out_dim=10, activation=nn.LeakyReLU):
        super().__init__()
   

        self.CNN=CNN_(image_shape=CNN_in_dim, out_dim=10,hidden=784)
        self.MLP=MLP_(in_dim=mlp_in_dim, out_dim=10)
        self.MLP_combine=MLP_(in_dim=20, out_dim=out_dim)


    def forward(self, x,y):
        
        '''
        x should have shape batch x channels x H x W 
        y should have shape batch x dim
        '''
        vec1=self.CNN(x)
        vec2=self.MLP(y)

        output=self.MLP_combine(torch.cat([vec1,vec2],1))

        return output