#from models.layer import ArmConv2d
#from models.layer.MAPConv2D import MAPConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
#from models.layer.MAPDense import MAPDense


import torch.optim as optim


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, weight_decay=0.,opt=None):
        super(BasicBlock, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        # self.conv1 = ArmConv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
        #                        droprate_init=droprate_init, weight_decay=weight_decay / (1 - 0.3),
        #                        local_rep=local_rep, lamba=lamba,opt=opt)

        # self.bn2 = nn.BatchNorm2d(out_planes)
        # self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
        #                        weight_decay=weight_decay)

        # self.equalInOut = (in_planes == out_planes)
        # self.convShortcut = (not self.equalInOut) and \
        #                     MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
        #                               weight_decay=weight_decay) or None


        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1,bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1,bias=False)

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,bias=False) or None


    def forward(self, x,z_mask_generator,mu_mask,mask_off):
        
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
            #x = F.relu (x)
        else:
            out = F.relu(self.bn1(x))
            #out = F.relu(x)
   
        out = self.conv1(out if self.equalInOut else x)

        ###mask after the first conv in each unit
        intermediate_state=out.detach().clone()

        mz_l,P_F_Z =z_mask_generator(out.detach())#detached from the computational graph of main model
        
        mz_l=mz_l
        m_mu_l=mu_mask
    
        if mask_off=="z_mask":  
            m=m_mu_l
        elif mask_off=="mu_mask":   
            m=mz_l
        elif mask_off=="none": 
            m=mz_l*m_mu_l
        elif mask_off=="both":
            m=torch.ones(mz_l.shape).to(device)
        
        out=out.mul(m.unsqueeze(2).unsqueeze(3)) 

        out = self.conv2(F.relu(self.bn2(out)))
        out=torch.add(out, x if self.equalInOut else self.convShortcut(x))
        #out = self.conv2(F.relu(out))
        return out,m,mz_l,P_F_Z




class ARMWideResNet_GFN(nn.Module):
    # droprate_init = 0.3
    def __init__(self, depth=28, num_classes=10, widen_factor=10, N=50000, beta_ema=0.99, weight_decay=5e-4,
                 lambas=0.001,opt=None):
        super(ARMWideResNet_GFN, self).__init__()
        
        self.opt=opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6 # 4
        self.N = N
        self.beta_ema = beta_ema
        self.epoch = 0
        self.elbo = torch.zeros(1)
        block = BasicBlock
        #droprate_init = self.opt.wrn_dr

        self.weight_decay = N * weight_decay
        self.lamba = lambas

        # 1st conv before any network block
        # self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
        #                        weight_decay=self.weight_decay)

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)


        #1st block [Conv + Dropout (160); Original Conv (160)] x 4
        self.blocks=nn.ModuleList()

        block1_pair1=BasicBlock(in_planes=nChannels[0], out_planes=nChannels[1],
         stride=1,
          weight_decay= self.weight_decay,
          opt=self.opt)
        
        self.blocks.append(block1_pair1)

        block1_pair2=BasicBlock(in_planes=nChannels[1], out_planes=nChannels[1],
         stride=1,
        weight_decay= self.weight_decay,
        opt=self.opt)

        self.blocks.append(block1_pair2)

        block1_pair3=BasicBlock(in_planes=nChannels[1], out_planes=nChannels[1],
         stride=1,
         weight_decay= self.weight_decay,
          opt=self.opt)

        self.blocks.append(block1_pair3)

        block1_pair4=BasicBlock(in_planes=nChannels[1], out_planes=nChannels[1],
         stride=1,
         weight_decay= self.weight_decay,
        opt=self.opt)

        self.blocks.append(block1_pair4)

        #2nd block [Conv + Dropout (320); Original Conv (320)] x 4
        block2_pair1=BasicBlock(in_planes=nChannels[1], out_planes=nChannels[2],
         stride=2,
        weight_decay= self.weight_decay,
          opt=self.opt)


        self.blocks.append(block2_pair1)

        block2_pair2=BasicBlock(in_planes=nChannels[2], out_planes=nChannels[2],
         stride=1,
         weight_decay= self.weight_decay,
          opt=self.opt)


        self.blocks.append(block2_pair2)

        block2_pair3=BasicBlock(in_planes=nChannels[2], out_planes=nChannels[2],
         stride=1,
         weight_decay= self.weight_decay,
          opt=self.opt)


        self.blocks.append(block2_pair3)


        block2_pair4=BasicBlock(in_planes=nChannels[2], out_planes=nChannels[2],
         stride=1,
         weight_decay= self.weight_decay,
         opt=self.opt)


        self.blocks.append(block2_pair4)


       #3rd block [Conv + Dropout (640); Original Conv (640)] x 4
        block3_pair1=BasicBlock(in_planes=nChannels[2], out_planes=nChannels[3],
         stride=2,
         weight_decay= self.weight_decay,
          opt=self.opt)


        self.blocks.append(block3_pair1)


        block3_pair2=BasicBlock(in_planes=nChannels[3], out_planes=nChannels[3],
         stride=1,
          weight_decay= self.weight_decay,
          opt=self.opt)

        self.blocks.append(block3_pair2)


        block3_pair3=BasicBlock(in_planes=nChannels[3], out_planes=nChannels[3],
         stride=1,
         weight_decay= self.weight_decay,
         opt=self.opt)

        self.blocks.append(block3_pair3)

        block3_pair4=BasicBlock(in_planes=nChannels[3], out_planes=nChannels[3],
         stride=1,
          weight_decay= self.weight_decay,
          opt=self.opt)


        self.blocks.append(block3_pair4)


   
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        #self.bn = nn.BatchNorm2d(nChannels[1])
        #self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)
        self.fcout = nn.Linear(nChannels[3], num_classes)
        #self.fcout = nn.Linear(2560, num_classes)

        # self.layers, self.bn_params, self.l0_layers = [], [], []
        # for m in self.modules():
        #     if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, ArmConv2d):
        #         self.layers.append(m)
        #         if isinstance(m, ArmConv2d):
        #             self.l0_layers.append(m)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #         self.bn_params += [m.weight, m.bias]
        # print('len',len(self.l0_layers))

        self.to(self.device)

        # if beta_ema > 0.:
        #     print('Using temporal averaging with beta: {}'.format(beta_ema))
        #     self.avg_param = deepcopy(list(p.data for p in self.parameters()))
        #     if torch.cuda.is_available():
        #         self.avg_param = [a.cuda() for a in self.avg_param]
        #     self.steps_ema = 0.

        # print('Using weight decay: {}'.format(self.weight_decay))

        


        ####GFN related
        maskgenerator_input_shapes=[(160,32,32)]*4+[(320,32,32)]+[(320,16,16)]*3+[(640,16,16)]+[(640,8,8)]*3
        hiddens=[16*7*7]*4+[16*7*7]+[16*3*3]*3+[16*3*3]+[16*1*1]*3

        # maskgenerator_input_shapes=[(160,32,32)]*4
        # hiddens=[16*7*7]*4

        # self.rand_mask_generators=construct_random_mask_generators(n_layers=len(maskgenerator_input_shapes),
        #                                                             dropout_rate=opt.dropout_rate).to(self.device)
        self.z_mask_generators=construct_z_mask_generators(image_shape=maskgenerator_input_shapes,
                                                        hiddens=hiddens).to(self.device)
    
        self.mu_mask_generators=construct_mu_mask_generators(image_shape=maskgenerator_input_shapes,
                                                        hidden=hiddens).to(self.device)

        self.LogZ_mu=nn.Parameter(torch.tensor(210.0))#start from a value closer to the magnitude of LogP_mu

        self.activation=nn.ReLU
        mg_activation=nn.LeakyReLU
        

        z_lr=1e-1
        mg_lr_z=1e-3
        mg_lr_mu=1e-3
        
        lr=1e-1
        self.beta=0.01 #temperature on rewards

        self.z_total_flowestimator = CNN_(image_shape=(3,32,32), out_dim=1,hidden=16*7*7,
                                            activation=nn.LeakyReLU).to(self.device)



        GFN_mu_param_list = [{'params': self.mu_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
        {'params': self.LogZ_mu, 'lr': z_lr,"weight_decay":0.1}]

        self.GFN_mu_optimizer = optim.Adam(GFN_mu_param_list)
       #self.GFN_mu_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.GFN_mu_optimizer, milestones=[200,300,350],gamma=0.2)

        GFN_z_param_list = [{'params': self.z_mask_generators.parameters(), 'lr': mg_lr_z,"weight_decay":0.1},
        {'params': self.z_total_flowestimator.parameters(), 'lr': z_lr,"weight_decay":0.1}]

        self.GFN_z_optimizer = optim.Adam(GFN_z_param_list)

        #self.GFN_z_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.GFN_z_optimizer, milestones=[200,300,350],gamma=0.2)

        taskmodel_param_list = [{'params': self.parameters(), 'lr': lr}]


        self.taskmodel_optimizer = optim.Adam(taskmodel_param_list)
        self.taskmodel_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.taskmodel_optimizer, milestones=self.opt.schedule_milestone,gamma=0.2)

        self.to(self.device)


    def forward(self, x, y=None):
        if self.train:

            logits,actual_masks,masks_mu,masks_z,LogZ_mu,LogPF_mu,LogR_mu,LogPB_mu,LogPF_BNN,LogZ_z,LogPF_z,LogR_z,LogPB_z  = self.GFN_forward(x)
        
        else:
            #sample multiple times
            repeat=10
            logits=[]
            for _ in range(repeat):
                logits_,actual_masks,masks_mu,masks_z,LogZ_mu,LogPF_mu,LogR_mu,LogPB_mu,LogPF_BNN,LogZ_z,LogPF_z,LogR_z,LogPB_z  = self.GFN_forward(x)
                logits.append(logits_.unsqueeze(2))
            logits=torch.logsumexp(torch.cat(logits,2),2)

        return logits


    #####GFN related functions
    def GFN_forward(self, x,mask_off="none"):

        batch_size,input_dim=x.shape[0],x.shape[1]
            
        LogZ_mu=self.LogZ_mu#top down mask has a separate Z and indepdent of input x
        LogZ_z=self.z_total_flowestimator(x)#partition function for z mask is input dependent
        
        LogPF_mu=torch.zeros(batch_size).to(device)# forward probability
        LogPB_mu=torch.zeros(batch_size).to(device)# backward prob
        LogR_mu=torch.zeros(batch_size).to(device)

        LogPF_BNN=torch.zeros(batch_size).to(device)
        LogPB_BNN=torch.zeros(batch_size).to(device)

        LogPF_z=torch.zeros(batch_size).to(device)# forward probability
        LogR_z=torch.zeros(batch_size).to(device)
        LogPB_z=torch.zeros(batch_size).to(device)


        '''
        mu mask generation, indepedent of the input x 
        '''

        #initialize masks as all zeros(dropout them all)
        #one batch share the same mu mask
        masks_mu=[[] for _ in range(len(self.blocks))]
        

        for layer_idx in range(len(self.blocks)):

                
            if layer_idx==0:

                mu_mask_l,mu_p_l=self.mu_mask_generators[layer_idx].to(device)(torch.zeros(batch_size,input_dim).to(device))      

                
                
            else:
                ##concatenate all previous masks
                previous_mask=[]
                for j in range(layer_idx):
                    previous_mask.append(masks_mu[j][-1])
                previous_mask=torch.cat(previous_mask,1).to(device)
                mu_mask_l,mu_p_l=self.mu_mask_generators[layer_idx].to(device)(previous_mask)
            
            masks_mu[layer_idx].append(mu_mask_l.detach().clone())
            

            
            LogPF_mu+=(mu_mask_l*torch.log(mu_p_l)+(1-mu_mask_l)*torch.log(1-mu_p_l)).sum(1)

            LogPB_mu+=0 #uniform prior
            


        '''
        forward pass
        '''

        actual_masks=[]
        masks_z=[]


        out = self.conv1(x)

        for layer_idx in range(len(self.blocks)):
            out,m,mz_l,P_F_Z=self.blocks[layer_idx].to(self.device)(out.to(self.device),self.z_mask_generators[layer_idx].to(self.device),
                                                    masks_mu[layer_idx][-1].to(self.device),mask_off)
            
            masks_z.append(mz_l)
            actual_masks.append(m)

            LogPF_z+=(mz_l*torch.log(P_F_Z)+(1-mz_l)*torch.log(1-P_F_Z)).sum(1)
            LogPB_z-=0
       

        out = F.relu(self.bn.to(self.device)(out))
        #out = F.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
     
        pred=self.fcout.to(self.device)(out)

 
        return pred,actual_masks,masks_mu,masks_z,LogZ_mu,LogPF_mu,LogR_mu,LogPB_mu,LogPF_BNN,LogZ_z,LogPF_z,LogR_z,LogPB_z



    def _gfn_step(self, x, y,mask_train="taskmodel",mask_off="none"):
        #####this step allows us to use different x,y to generate mask and calcualte reward(loss)

 
        metric = {}
        logits,actual_masks,masks_mu,masks_z,LogZ_mu,LogPF_mu,LogR_mu,LogPB_mu,LogPF_BNN,LogZ_z,LogPF_z,LogR_z,LogPB_z = self.GFN_forward(x,mask_off)
        
        #loss calculation
        CEloss = nn.CrossEntropyLoss(reduction='none')(logits, y)
        TaskLoss=CEloss
        
        ###the original math fomula should be LogR_my= *Data_set_size, but we use a fix beta here to adjust peakiness of the distribution
        LogR_mu-=self.beta*TaskLoss.detach().clone()#smaller beta makes the reward less peaky   

        LogR_z-=self.beta*TaskLoss.detach().clone()#not times by N for z mask generator   

        GFN_loss_mu=(LogZ_mu+LogPF_mu-LogR_mu-LogPB_mu)**2

        GFN_loss_z=(LogZ_z+LogPF_z-LogR_z-LogPB_z)**2 #GFN for z mask to be trained separately


        # Update model

        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
        
        metric['CELoss'] = CEloss.mean().item()#loss output from the model
     

 
        metric['GFNloss_mu'] = GFN_loss_mu.mean().item()#loss output from the model is per sample
        metric['GFNloss_z'] = GFN_loss_z.mean().item()#loss output from the model is per sample
        
        metric['acc'] = acc

        if mask_train=="taskmodel": 

            taskmodel_loss=CEloss.mean()

            self.taskmodel_optimizer.zero_grad()

            taskmodel_loss.backward()
            
            self.taskmodel_optimizer.step()


        if mask_train=="mu_mask": 
            self.GFN_mu_optimizer.zero_grad()

            GFN_loss_mu.sum().backward()
            
            self.GFN_mu_optimizer.step()
        

        elif mask_train=="z_mask": 
            
            self.GFN_z_optimizer.zero_grad()

            GFN_loss_z.sum().backward()
            
            self.GFN_z_optimizer.step()
        
        ###calculated actual droppout rate
        actual_dropout_rate=0

        n_units=0
        n_dropped=0.0
        batch_size=x.shape[0]
        for layer_idx in range(len(self.blocks)):
            m=actual_masks[layer_idx]
            n_units+=m.shape[1]
            
            n_dropped+=(m==0).float().mean(0).sum()

        actual_dropout_rate=n_dropped/n_units 
        metric['actual_dropout_rate']=actual_dropout_rate.item()


        #differnet terms of TB
        metric['LogZ_mu']=LogZ_mu.mean().item()
        
        metric['LogPF_mu']=LogPF_mu.mean().item()

        metric['LogR_mu']=LogR_mu.mean().item()
        
        metric['LogPB_mu']=LogPB_mu.mean().item()


        metric['LogZ_z']=LogZ_z.mean().item()
        
        metric['LogPF_z']=LogPF_z.mean().item()

        metric['LogR_z']=LogR_z.mean().item()
        
        metric['LogPB_z']=LogPB_z.mean().item()



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


def construct_z_mask_generators(
        image_shape,
        hiddens,
        activation=nn.LeakyReLU
):

    mask_generators = nn.ModuleList()
    for input_size,hid_dim in zip(image_shape,hiddens):

        mask_generators.append(
            CNNMaskGenerator(
                input_shape=input_size,
                hidden=hid_dim,
                activation=activation
            ).to(device)
        )

    return mask_generators



def construct_mu_mask_generators(
        image_shape,
        hidden=None,
        activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()

    for layer_idx in range(len(image_shape)):

        if layer_idx==0:
            in_dim=3
            out_dim=image_shape[layer_idx][0]
        
        else:
            in_dim=0
            for j in range(layer_idx):
                 in_dim+=image_shape[j][0]

            out_dim=image_shape[layer_idx][0]

        
        mask_generators.append(
            MLPMaskGenerator(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden=hidden,
                activation=activation
            ).to(device)
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