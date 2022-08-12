import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

import torchbnn as bnn
import random


####BNN version


class RandomMaskGenerator(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)

    def forward(self, x):
        return torch.bernoulli((1. - self.dropout_rate) * torch.ones(x.shape))

    def log_prob(self, x, m):
        dist = (1. - self.dropout_rate) * torch.ones(x.shape)
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


def construct_random_mask_generators(
        model,
        dropout_rate,
):
    mask_generators = nn.ModuleList()
    for layer in model.fc:
        mask_generators.append(
            RandomMaskGenerator(
                dropout_rate=dropout_rate,
            )
        )

    return mask_generators


def construct_mlp_mu_mask_generators(
        model,
        hidden=None,
        activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()
    for layer_idx in range(len(model.fc)):
        #also drop out on the input
        if layer_idx==0:#input dropout
            in_dim=model.fc[0].weight_mu.shape[1]
            out_dim=model.fc[0].weight_mu.shape[0]
        
        else:
            in_dim=0
            for j in range(layer_idx):
                 in_dim+=model.fc[j].weight_mu.shape[0]

            out_dim=model.fc[layer_idx].weight_mu.shape[0]

        
        mask_generators.append(
            MLPMaskGenerator(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden=hidden,
                activation=activation
            )
        )

    return mask_generators






class BaysianMLPMaskedDropout(nn.Module):

    def __init__(self, in_dim=784, out_dim=10, hidden=None, activation=nn.LeakyReLU,beta=1.0):
        super().__init__()
        if hidden is None:
            hidden = [32, 32]
        h_old = in_dim
        self.fc = nn.ModuleList()
        for h in hidden:
            self.fc.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,in_features=h_old, out_features=h))
            h_old = h
        self.out_layer = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,in_features=h_old, out_features=out_dim)

        self.activation = activation

        self.beta=beta

    def forward(self, x,y, mu_mask_generators,LogZ_mu,Data_set_size,mask_off=False,epsilon=0):
        batch_size,input_dim=x.shape[0],x.shape[1]


            
        LogZ_mu=LogZ_mu#top down mask has a separate Z and indepdent of input x

        LogPF_mu=torch.zeros(batch_size)# forward probability
        LogPB_mu=torch.zeros(batch_size)# backward prob
        LogR_mu=torch.zeros(batch_size)

        LogPF_BNN=torch.zeros(batch_size)
        '''
        mu mask generation, indepedent of the input x 
        '''

        #initialize masks as all zeros(dropout them all)
        #one batch share the same mu mask
        masks_mu=[[] for _ in range(len(self.fc))]
        

        for layer_idx in range(len(self.fc)):

                
            if layer_idx==0:
                mu_mask_l,mu_p_l=mu_mask_generators[layer_idx](torch.zeros(batch_size,input_dim))      
                
            else:
                ##concatenate all previous masks
                previous_mask=[]
                for j in range(layer_idx):
                    previous_mask.append(masks_mu[j][-1])
                previous_mask=torch.cat(previous_mask,1)
                mu_mask_l,mu_p_l=mu_mask_generators[layer_idx](previous_mask)
            
            masks_mu[layer_idx].append(mu_mask_l.detach().clone())
            

            
            LogPF_mu+=(mu_mask_l*torch.log(mu_p_l)+(1-mu_mask_l)*torch.log(1-mu_p_l)).sum(1)

            LogPB_mu+=0 #uniform prioe
            

        '''
        forward pass
        '''
        ####dropout on inputs

        for layer_idx in range(len(self.fc)):
            layer=self.fc[layer_idx]


     
            m_mu_l=masks_mu[layer_idx][-1]

            if epsilon>0 and random.uniform(0, 1)<epsilon:
                m_mu_l=torch.bernoulli(torch.zeros(m_mu_l.shape).fill_(0.5))#generate random masks                
            ###occasionally use 
            m=m_mu_l


            ###apply the task model
            weight_ = layer.weight_mu + torch.exp(layer.weight_log_sigma) * torch.randn_like(layer.weight_log_sigma) 
            bias_ = layer.bias_mu + torch.exp(layer.bias_log_sigma) * torch.randn_like(layer.bias_log_sigma)

            x=F.linear(x, weight_, bias_)

            x = self.activation()(x)
            
            if not mask_off:
                multipliers = m.shape[1] / (m.sum(1) + 1e-6)
                x = torch.mul((x * m).T, multipliers).T

            ####probabilitty of picking Bayesian neural network 
            log_pdf_weights=torch.distributions.normal.Normal(layer.weight_mu,torch.exp(layer.weight_log_sigma)).log_prob(weight_)  
            
            # if layer_idx==1 and random.uniform(0, 1)<0.01:
            #     print(layer.weight_mu)

            log_pdf_bias=torch.distributions.normal.Normal(layer.bias_mu,torch.exp(layer.bias_log_sigma)).log_prob(bias_)
            
            ###add P_F_wb to the GFN loss
            LogPF_BNN+=(log_pdf_weights.sum()+log_pdf_bias.sum()).repeat(batch_size)

        ####output final prediction
        pred = self.out_layer(x)

        #### negative log likelihood loss
        CEloss = nn.CrossEntropyLoss(reduction='none')(pred, y)

        TaskLoss=Data_set_size*CEloss

        LogR_mu-=self.beta*TaskLoss#.detach().clone()#smaller beta makes the reward less peaky


         
        
        
        GFN_loss_mu=(LogZ_mu+LogPF_mu+LogPF_BNN-LogR_mu-LogPB_mu)**2



        return pred,masks_mu,CEloss,GFN_loss_mu,LogZ_mu,LogPF_mu,LogR_mu,LogPB_mu,LogPF_BNN

class BaysianMLPClassifierWithMaskGenerator(nn.Module):
    def __init__(
            self,
            in_dim=784,
            out_dim=10,
            hidden=None,
            activation=nn.LeakyReLU,
            dropout_rate=0.5,
            mg_type='gfn',
            lr=1e-3,
            z_lr=1e-2,
            mg_lr=1e-3,
            mg_hidden=None,
            mg_activation=nn.LeakyReLU,
            beta=0.1,
            device='cpu',
    ):
        super().__init__()
        
        # classifier
        self.model = BaysianMLPMaskedDropout(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            activation=activation,
            beta=beta,
        ).to(device)

        # mask generators
        self.mg_type = mg_type
        if mg_type == 'random':
            self.mu_mask_generators = construct_random_mask_generators(
                model=self.model,
                dropout_rate=dropout_rate
            ).to(device)

            self.z_mask_generators = construct_random_mask_generators(
            model=self.model,
            dropout_rate=dropout_rate
            ).to(device)
        elif mg_type == 'gfn':
            # for backward log prob calculation only
            self.rand_mask_generators = construct_random_mask_generators(
                model=self.model,
                dropout_rate=dropout_rate
            ).to(device)

            self.mu_mask_generators = construct_mlp_mu_mask_generators(
            model=self.model,
            hidden=mg_hidden,
            activation=mg_activation,
            ).to(device)


            self.LogZ_mu=nn.Parameter(torch.tensor(0.0))

            # GFN_mu_param_list = [{'params': self.mu_mask_generators.parameters(), 'lr': mg_lr,"weight_decay":0.1},
            #             {'params': self.model.LogZ_mu, 'lr': z_lr,"weight_decay":0.1},]


            GFN_mu_param_list = [{'params': self.mu_mask_generators.parameters(), 'lr': mg_lr,"weight_decay":0.1},
                            {'params': self.model.parameters(), 'lr': lr,"weight_decay":0.1},
                            {'params': self.LogZ_mu, 'lr': z_lr,"weight_decay":0.1}]

            self.GFN_mu_optimizer = optim.Adam(GFN_mu_param_list)


            BNN_param_list = [{'params': self.model.parameters(), 'lr': lr}]


            self.BNN_optimizer = optim.Adam(BNN_param_list)


        else:
            raise ValueError('unknown mask generator type {}'.format(mg_type))




    def BNNstep(self, x, y,Data_set_size,mask_off=True):
        ####this step only trains the BNN


        metric = {}
        logits,masks_mu,CEloss,GFN_loss_mu,LogZ_mu,LogPF_mu,LogR_mu,LogPB_mu,LogPF_BNN = self.model(x,y, 
                                                    self.mu_mask_generators,self.LogZ_mu,
                                                    Data_set_size,
                                                    mask_off)
        # Update model


        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
        
        metric['CELoss'] = CEloss.mean().item()#loss output from the model
        
        kl_weight = 0.1

        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

        kl = kl_loss(self.model)

        metric['kl_loss']=kl.item()

        BNN_loss=CEloss.mean()+kl_weight*kl#loss only for BNN


        metric['GFNloss_mu'] = GFN_loss_mu.mean().item()#loss output from the model is per sample
        
        metric['acc'] = acc

        self.BNN_optimizer.zero_grad()

        BNN_loss.backward()
        
        self.BNN_optimizer.step()

        
        ###calculated actual droppout rate
        actual_dropout_rate=0

        n_units=0
        n_dropped=0.0
        batch_size=x.shape[0]
        for layer_idx in range(len(self.model.fc)):
            m=masks_mu[layer_idx][-1]
            n_units+=m.shape[1]
            
            n_dropped+=(m==0).float().mean(0).sum()

        actual_dropout_rate=n_dropped/n_units 
        metric['actual_dropout_rate']=actual_dropout_rate


        #differnet terms of TB
        metric['LogZ_mu']=LogZ_mu.mean().item()
        
        metric['LogPF_mu']=LogPF_mu.mean().item()
 
        metric['LogPF_BNN']=LogPF_BNN.mean().item()
                
        metric['LogR_mu']=LogR_mu.mean().item()
        
        metric['LogPB_mu']=LogPB_mu.mean().item()



        return metric



    def GFNstep(self, x, y,Data_set_size,mask_off=False,epsilon=0.0):
        ####train GFN 


        metric = {}
        logits,masks_mu,CEloss,GFN_loss_mu,LogZ_mu,LogPF_mu,LogR_mu,LogPB_mu,LogPF_BNN = self.model(x,y, 
                                                    self.mu_mask_generators,self.LogZ_mu,
                                                    Data_set_size,
                                                    mask_off,epsilon)
        # Update model


        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
        
        metric['CELoss'] = CEloss.mean().item()#loss output from the model
        
        kl_weight = 0.1

        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

        kl = kl_loss(self.model)

        metric['kl_loss']=kl.item()

        BNN_loss=CEloss.mean()+kl_weight*kl#loss only for BNN


 
        metric['GFNloss_mu'] = GFN_loss_mu.mean().item()#loss output from the model is per sample
        
        metric['acc'] = acc

        self.GFN_mu_optimizer.zero_grad()

        GFN_loss_mu.sum().backward()
        
        self.GFN_mu_optimizer.step()

     
        ###calculated actual droppout rate
        actual_dropout_rate=0

        n_units=0
        n_dropped=0.0
        batch_size=x.shape[0]
        for layer_idx in range(len(self.model.fc)):
            m=masks_mu[layer_idx][-1]
            n_units+=m.shape[1]
            
            n_dropped+=(m==0).float().mean(0).sum()

        actual_dropout_rate=n_dropped/n_units 
        metric['actual_dropout_rate']=actual_dropout_rate


        #differnet terms of TB
        metric['LogZ_mu']=LogZ_mu.mean().item()
        
        metric['LogPF_mu']=LogPF_mu.mean().item()

        metric['LogPF_BNN']=LogPF_BNN.mean().item()
        
        metric['LogR_mu']=LogR_mu.mean().item()
        
        metric['LogPB_mu']=LogPB_mu.mean().item()
        return metric


    def test(self, x, y,Data_set_size,masks_off=False,epsilon=0,SampleRepeats=10):
        metric = {}
        SampleRepeats=10
        batch_size=x.shape[0]
        with torch.no_grad():
            logits=[]
            masks=[]
            for _ in range(SampleRepeats):
                
                logits_,masks_mu,CEloss,GFN_loss_mu,LogZ_mu,LogPF_mu,LogR_mu,LogPB_mu,LogPF_BNN=self.model(x,y, 
                                            self.mu_mask_generators,self.LogZ_mu,
                                            Data_set_size,masks_off,epsilon)###y is used for loss caculation but not prediction
                logits.append(logits_.unsqueeze(2))

                masks_=[]
                for layer_idx in range(len(self.model.fc)):
                    m=masks_mu[layer_idx][-1]
                    masks_.append(m)

                masks_=torch.cat(masks_,1)
                masks.append(masks_.unsqueeze(2))


            
            logits=torch.logsumexp(torch.cat(logits,2),2)

            
            masks=torch.cat(masks,2)


              
            loss = nn.CrossEntropyLoss()(logits, y)
            acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
            metric['CELoss'] = loss.item()
            metric['acc'] = acc

            metric['GFNloss_mu'] = GFN_loss_mu.mean().item()#loss output from the model is per sample

            #differnet terms of TB
            metric['LogZ_mu']=LogZ_mu.mean().item()

            metric['LogPF_mu']=LogPF_mu.mean().item()

            metric['LogPF_BNN']=LogPF_BNN.mean().item()


            metric['LogR_mu']=LogR_mu.mean().item()

            metric['LogPB_mu']=LogPB_mu.mean().item()


            ###calculated actual droppout rate
            actual_dropout_rate=0

            n_units=0
            n_dropped=0.0
            batch_size=x.shape[0]
            for layer_idx in range(len(self.model.fc)):
                m=masks_mu[layer_idx][-1]
                n_units+=m.shape[1]

                n_dropped+=(m==0).float().mean(0).sum()

            actual_dropout_rate=n_dropped/n_units 
            metric['actual_dropout_rate']=actual_dropout_rate


            #for plotting
            LogZ_mu_individual=LogZ_mu
            Log_PF_mu_individual=LogPF_mu

            Log_PF_BNN_individual=LogPF_BNN

            R_mu_individual=LogR_mu
           
        return metric,masks,LogZ_mu_individual,Log_PF_mu_individual,R_mu_individual,Log_PF_BNN_individual
