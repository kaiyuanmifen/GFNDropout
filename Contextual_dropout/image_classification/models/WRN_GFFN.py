from models.layer import ArmConv2d
from models.layer.MAPConv2D import MAPConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from models.layer.MAPDense import MAPDense


import torch.optim as optim

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, local_rep=False,opt=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = ArmConv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                               droprate_init=droprate_init, weight_decay=weight_decay / (1 - 0.3),
                               local_rep=local_rep, lamba=lamba,opt=opt)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None

    def forward(self, x,mask_generator):
    
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
            #x = F.relu (x)
        else:
            out = F.relu(self.bn1(x))
            #out = F.relu(x)
   
        out = self.conv1(out if self.equalInOut else x)

        ###mask after the first conv in each unit
        intermediate_state=out.detach().clone()
        m=mask_generator(out).detach().unsqueeze(2).unsqueeze(3)
        out=out.mul(m) 

        out = self.conv2(F.relu(self.bn2(out)))
        #out = self.conv2(F.relu(out))
        return torch.add(out, x if self.equalInOut else self.convShortcut(x)),m,intermediate_state




class ARMWideResNet_GFFN(nn.Module):
    # droprate_init = 0.3
    def __init__(self, depth=28, num_classes=10, widen_factor=10, N=50000, beta_ema=0.99, weight_decay=5e-4,
                 lambas=0.001,opt=None):
        super(ARMWideResNet_GFFN, self).__init__()
        
        self.opt=opt

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6 # 4
        self.N = N
        self.beta_ema = beta_ema
        self.epoch = 0
        self.elbo = torch.zeros(1)
        block = BasicBlock
        droprate_init = self.opt.wrn_dr

        self.weight_decay = N * weight_decay
        self.lamba = lambas

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)

        #1st block [Conv + Dropout (160); Original Conv (160)] x 4
        self.block1_pair1=BasicBlock(in_planes=nChannels[0], out_planes=nChannels[1],
         stride=1,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)

        self.block1_pair2=BasicBlock(in_planes=nChannels[1], out_planes=nChannels[1],
         stride=1,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)

        self.block1_pair3=BasicBlock(in_planes=nChannels[1], out_planes=nChannels[1],
         stride=1,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)

        self.block1_pair4=BasicBlock(in_planes=nChannels[1], out_planes=nChannels[1],
         stride=1,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)


        #2nd block [Conv + Dropout (320); Original Conv (320)] x 4
        self.block2_pair1=BasicBlock(in_planes=nChannels[1], out_planes=nChannels[2],
         stride=2,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)

        self.block2_pair2=BasicBlock(in_planes=nChannels[2], out_planes=nChannels[2],
         stride=1,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)

        self.block2_pair3=BasicBlock(in_planes=nChannels[2], out_planes=nChannels[2],
         stride=1,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)

        self.block2_pair4=BasicBlock(in_planes=nChannels[2], out_planes=nChannels[2],
         stride=1,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)

       #3rd block [Conv + Dropout (640); Original Conv (640)] x 4
        self.block3_pair1=BasicBlock(in_planes=nChannels[2], out_planes=nChannels[3],
         stride=2,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)

        self.block3_pair2=BasicBlock(in_planes=nChannels[3], out_planes=nChannels[3],
         stride=1,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)

        self.block3_pair3=BasicBlock(in_planes=nChannels[3], out_planes=nChannels[3],
         stride=1,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)

        self.block3_pair4=BasicBlock(in_planes=nChannels[3], out_planes=nChannels[3],
         stride=1,
         droprate_init=droprate_init, weight_decay= self.weight_decay,
          lamba=self.lamba, local_rep=self.opt.local_rep,opt=self.opt)



   
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)

        self.layers, self.bn_params, self.l0_layers = [], [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, ArmConv2d):
                self.layers.append(m)
                if isinstance(m, ArmConv2d):
                    self.l0_layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]
        print('len',len(self.l0_layers))

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))

        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        ####GFN related
        maskgenerator_input_shapes=[(160,32,32)]*4+[(320,32,32)]+[(320,16,16)]*3+[(640,16,16)]+[(640,8,8)]*3
        hiddens=[16*7*7]*4+[16*7*7]+[16*3*3]*3+[16*3*3]+[16*1*1]*3

        self.rand_mask_generators=construct_random_mask_generators(n_layers=len(maskgenerator_input_shapes),
                                                                    dropout_rate=opt.dropout_rate)
        self.mask_generators=construct_mask_generators(image_shape=maskgenerator_input_shapes,
                                                        hiddens=hiddens,
                                                        dropout_rate=opt.dropout_rate)
        self.activation=nn.ReLU
        mg_activation=nn.LeakyReLU
        z_lr=1e-1
        mg_lr=1e-3
        self.total_flowestimator = CNN_GFFN(image_shape=(3,32,32), out_dim=1,hidden=16*7*7,
                                            activation=nn.LeakyReLU)

        MaskGeneratorParameters=[]
        for generator in self.mask_generators:
            MaskGeneratorParameters+=list(generator.parameters())

        param_list = [{'params': MaskGeneratorParameters, 'lr': mg_lr},
                     {'params': self.total_flowestimator.parameters(), 'lr': z_lr}]
        

        self.mg_optimizer = optim.Adam(param_list)



        self.to(self.device)


    def forward(self, x, y=None):
        if self.train:

            score,_,_ = self.GFFN_forward(x)
        else:
            #sample multiple times
            repeat=10
            score=[]
            for _ in range(repeat):
                score_,_,_ = self.GFFN_forward(x)
                score.append(score_.unsqueeze(2))
            score=torch.cat(score,2).mean(2)
        return score


    #####GFFFN related functions
    def GFFN_forward(self, x):
        masks=[]
        intermediate_states=[]#for TB/DB matching

        out = self.conv1(x)


        ####block1
        out,m,ims=self.block1_pair1(out,self.mask_generators[0])
        intermediate_states.append(ims)
        masks.append(m)

        out,m,ims=self.block1_pair2(out,self.mask_generators[1])
        intermediate_states.append(ims)
        masks.append(m)

        out,m,ims=self.block1_pair3(out,self.mask_generators[2])
        intermediate_states.append(ims)
        masks.append(m)

        out,m,ims=self.block1_pair4(out,self.mask_generators[3])
        intermediate_states.append(ims)
        masks.append(m)


        ####block2
        out,m,ims=self.block2_pair1(out,self.mask_generators[4])
        intermediate_states.append(ims)
        masks.append(m)

        out,m,ims=self.block2_pair2(out,self.mask_generators[5])
        intermediate_states.append(ims)
        masks.append(m)

        out,m,ims=self.block2_pair3(out,self.mask_generators[6])
        intermediate_states.append(ims)
        masks.append(m)

        out,m,ims=self.block2_pair4(out,self.mask_generators[7])
        intermediate_states.append(ims)
        masks.append(m)        

        ####block3
        out,m,ims=self.block3_pair1(out,self.mask_generators[8])
        intermediate_states.append(ims)
        masks.append(m)

        out,m,ims=self.block3_pair2(out,self.mask_generators[9])
        intermediate_states.append(ims)
        masks.append(m)

        out,m,ims=self.block3_pair3(out,self.mask_generators[10])
        intermediate_states.append(ims)
        masks.append(m)

        out,m,ims=self.block3_pair4(out,self.mask_generators[11])
        intermediate_states.append(ims)
        masks.append(m)      

        out = F.relu(self.bn(out))
        #out = F.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out), masks,intermediate_states


    def GFFN_forward_predefinedmasks(self, x,predefined_masks):
        print("to to defined")



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


        beta=1#temperature
        with torch.no_grad():
            losses = nn.CrossEntropyLoss(reduce=False)(logits, y_reward)
            log_rewards = - beta * losses
            logZ=self.total_flowestimator(x_mask)#this flow is calculated using x_mask, not a bug , to encourage generalization 
        # trajectory balance loss
        log_probs_F = []
        log_probs_B = []
        for m, mg_f, mg_b,ims in zip(masks, self.mask_generators, self.rand_mask_generators,intermediate_states):
            m=m.squeeze(3).squeeze(2)
            log_probs_F.append(mg_f.log_prob(ims, m).unsqueeze(1))
            log_probs_B.append(mg_b.log_prob(m, m).unsqueeze(1))
    
        tb_loss = ((logZ - log_rewards
                    + torch.cat(log_probs_F, dim=1).sum(dim=1)
                    - torch.cat(log_probs_B, dim=1).sum(dim=1)) ** 2).mean()
        metric['tb_loss'] = tb_loss.item()
        self.mg_optimizer.zero_grad()
        tb_loss.backward()
        self.mg_optimizer.step()


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



class CNNMaskGenerator(nn.Module):
    def __init__(self, input_shape,dropout_rate, hidden, activation=nn.LeakyReLU):
        super().__init__()
        self.num_unit = torch.tensor(input_shape[0]).type(torch.float32)
        self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)

        self.CNN= CNN_GFFN(
            image_shape=input_shape,
            out_dim=input_shape[0],
            activation=activation,
            hidden=hidden,
        )

    def _dist(self, x):
        x = self.CNN(x)
 
        x = torch.sigmoid(x)
        dist = (1. - self.dropout_rate) * self.num_unit * x / (x.sum(1).unsqueeze(1) + 1e-6)
        dist = dist.clamp(0, 1)
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
    for layer in range(n_layers):
        mask_generators.append(
            RandomMaskGenerator(
                dropout_rate=dropout_rate,
            )
        )

    return mask_generators


def construct_mask_generators(
        image_shape,
        dropout_rate,
        hiddens,
        activation=nn.LeakyReLU
):

    mask_generators = nn.ModuleList()
    for input_size,hid_dim in zip(image_shape,hiddens):

        mask_generators.append(
            CNNMaskGenerator(
                input_shape=input_size,
                dropout_rate=dropout_rate,
                hidden=hid_dim,
                activation=activation
            )
        )

    return mask_generators



class MLP_GFFN(nn.Module):
    def __init__(self, in_dim=784, out_dim=10, hidden=None, activation=nn.LeakyReLU):
        super().__init__()
        if hidden is None:
            hidden = [32, 32]

        self.LN=nn.LayerNorm(in_dim)##normalize the input to prevent exploding
        h_old = in_dim
        self.fc = nn.ModuleList()
        for h in hidden:
            self.fc.append(nn.Linear(h_old, h))
            h_old = h
        self.out_layer = nn.Linear(h_old, out_dim)
        self.activation = activation

    def forward(self, x):
        x=self.LN(x)
        for layer in self.fc:
            x = self.activation()(layer(x))
        x = self.out_layer(x)
        return x




class CNN_GFFN(nn.Module):
    def __init__(self, image_shape=(160,32,32), out_dim=10, hidden=16*5*5, activation=nn.LeakyReLU):
        super().__init__()
   

        self.LN=nn.LayerNorm(image_shape)##normalize the input to prevent exploding
        
        n_in_channels=image_shape[0]
        self.conv1 = nn.Conv2d(n_in_channels, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.fc1 = nn.Linear(hidden, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)
        self.activation = activation

    def forward(self, x):
        x=self.LN(x)
        x = self.pool(self.activation()(self.conv1(x)))
        
        x = self.pool(self.activation()(self.conv2(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        

        x = self.activation()(self.fc1(x))
        x = self.activation()(self.fc2(x))
        x = self.fc3(x)

        return x