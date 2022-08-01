import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

import torchbnn as bnn



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


class MLPMaskGenerator(nn.Module):
    def __init__(self, in_dim,out_dim, hidden=None, activation=nn.LeakyReLU):
        super().__init__()
       

        self.mlp = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            activation=activation,
        )

    def _dist(self, x):
        x = self.mlp(x)
        x = torch.sigmoid(x)
   
        dist=x
        #dist = (1. - self.dropout_rate) * self.num_unit * x / (x.sum(1).unsqueeze(1) + 1e-6)
        #dist = dist.clamp(0, 1)
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
        
        if layer_idx==(len(model.fc)-1):
            in_dim=model.fc[layer_idx].weight_mu.shape[0]+model.fc[layer_idx].weight_mu.shape[1]+model.out_layer.weight_mu.shape[1]
        else:
            in_dim=model.fc[layer_idx].weight_mu.shape[0]+model.fc[layer_idx].weight_mu.shape[1]+model.fc[layer_idx+1].weight_mu.shape[1]
        
        out_dim=model.fc[layer_idx].weight_mu.shape[1]

        
        mask_generators.append(
            MLPMaskGenerator(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden=hidden,
                activation=activation
            )
        )

    return mask_generators



def construct_mu_backwards_policy(
        model,
        hidden=None,
        activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()
    for layer_idx in range(len(model.fc)):

        if layer_idx==(len(model.fc)-1):
            in_dim=model.fc[layer_idx].weight_mu.shape[0]+model.fc[layer_idx].weight_mu.shape[1]+model.out_layer.weight_mu.shape[1]
            
            out_dim=model.fc[layer_idx].weight_mu.shape[0]+model.fc[layer_idx].weight_mu.shape[1]+model.out_layer.weight_mu.shape[1]
        
        else:     
            in_dim=model.fc[layer_idx].weight_mu.shape[0]+model.fc[layer_idx].weight_mu.shape[1]+model.fc[layer_idx+1].weight_mu.shape[1]
        
            out_dim=model.fc[layer_idx].weight_mu.shape[0]+model.fc[layer_idx].weight_mu.shape[1]+model.fc[layer_idx+1].weight_mu.shape[1]
        
        mask_generators.append(
            MLPMaskGenerator(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden=hidden,
                activation=activation
            )
        )

    return mask_generators


def construct_mlp_z_mask_generators(
        model,
        hidden=None,
        activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()
    for layer in model.fc:
        mask_generators.append(
            MLPMaskGenerator(
                in_dim=layer.weight_mu.shape[0],
                out_dim=layer.weight_mu.shape[1],
                hidden=hidden,
                activation=activation
            )
        )

    return mask_generators


class BaysianMLPMaskedDropout(nn.Module):

    def __init__(self, in_dim=784, out_dim=10, hidden=None, activation=nn.LeakyReLU):
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

    def forward(self, x,y, mu_mask_generators,z_mask_generators,mu_P_b,total_flowestimator,Data_set_size):
        batch_size,input_dim=x.shape[0],x.shape[1]
        intermediates=[]

        #this loss of the batch = (logz + logP_F -logP_B -R)**2 where R = prediction loss plus weight penalization
        Z=total_flowestimator(x)#each x has a different Z
        batch_GFN_loss=Z

        '''
        mu mask generation, indepedent of the input x 
        '''

        #initialize masks as all zeros(dropout them all)
        #one batch share the same mu mask
        masks_mu=[[] for _ in range(len(self.fc))]
        for layer_idx in range(len(self.fc)):
            masks_mu[layer_idx].append(torch.zeros(self.fc[layer_idx].weight_mu.shape[1]))
            #masks_mu=torch.ones(self.fc[layer_idx].weight_mu.shape[1])

        #generate the mu mask  and mask of next layer
        K=3 #number of computational iterations
        for k in range(1,K+1):
            for layer_idx in range(len(self.fc)):
                
                if layer_idx==0:
                    m_lm1_km1=torch.zeros(self.fc[layer_idx].weight_mu.shape[0])
                else:
                    m_lm1_km1=masks_mu[layer_idx-1][k-1]#conditioned on mask from previous layer
                
                m_l_km1=masks_mu[layer_idx][k-1]#mask of current layer 
                
                if layer_idx==(len(self.fc)-1):
                    m_lp1_km1=torch.zeros(self.fc[layer_idx].weight_mu.shape[1])
                else:
                    m_lp1_km1=masks_mu[layer_idx+1][k-1]#conditioned on mask from next layer
                
                m_l_k,p_l_k=mu_mask_generators[layer_idx](torch.cat([m_lm1_km1,m_l_km1,m_lp1_km1]))
                masks_mu[layer_idx].append(m_l_k)#mask mu_l_k
                
                ##add logP_F to GFN loss
                batch_GFN_loss+=(m_l_k*torch.log(p_l_k)-(1-m_l_k)*torch.log(1-p_l_k)).sum()

  
            ##minus logP_F to GFN loss
            for layer_idx in range(len(self.fc)):
                
                #masks of previou step 
                if layer_idx==0:
                    m_lm1_km1=torch.zeros(self.fc[layer_idx].weight_mu.shape[0])
                else:
                    m_lm1_km1=masks_mu[layer_idx-1][k-1]#conditioned on mask from previous layer
                
                m_l_km1=masks_mu[layer_idx][k-1]#mask of current layer 
                
                if layer_idx==(len(self.fc)-1):
                    m_lp1_km1=torch.zeros(self.fc[layer_idx].weight_mu.shape[1])
                else:
                    m_lp1_km1=masks_mu[layer_idx+1][k-1]#conditioned on mask from next layer
                

                #masks of current step 
                if layer_idx==0:
                    m_lm1_k=torch.zeros(self.fc[layer_idx].weight_mu.shape[0])
                else:
                    m_lm1_k=masks_mu[layer_idx-1][k]#conditioned on mask from previous layer
                
                m_l_k=masks_mu[layer_idx][k]#mask of current layer 
                
                if layer_idx==(len(self.fc)-1):
                    m_lp1_k=torch.zeros(self.fc[layer_idx].weight_mu.shape[1])
                else:
                    m_lp1_k=masks_mu[layer_idx+1][k]#conditioned on mask from next layer
                
                q_l_km1=mu_P_b[layer_idx](torch.cat([m_lm1_k,m_l_k,m_lp1_k]))

                batch_GFN_loss-=(m_l_km1*torch.log(q_l_km1)-(1-m_l_km1)*torch.log(1-q_l_km1)).sum()

            

        '''
        generate z mask , which are input dependent
        and pass in the input
        '''
        masks_z=[]
        for layer_idx in range(len(self.fc)):
            layer=self.fc[layer_idx]

            mz_l = z_mask_generators[layer_idx](x)#generate mask based on activation from previous layer
            masks_z.append(x)

            ####add log P_F_Z to the GFN loss

            batch_GFN_loss+=(mz_l*torch.log(P_F_Z)+(1-mz_l)*torch.log(1-P_F_Z)).sum(1)

            ###minize log P_B_z

            batch_GFN_loss-=0 # as it is a tree structure, back policy p_B_Z=1 and log(p_B_Z)=0



            
            ###combine mu and z masks
            m_mu_l=masks_mu[layer_idx][-1].unsqueeze(0).repeat(batch_size,1)#same mu mask for all sample in a batch


            #####combine masks
            m=mz_l*m_mu_l#multiply the mu and z masks to obtain final mask
            

            ###apply the task model
            weight_ = layer.weight_mu + torch.exp(layer.weight_log_sigma) * torch.randn_like(layer.weight_log_sigma) 
            bias_ = layer.bias_mu + torch.exp(layer.bias_log_sigma) * torch.randn_like(layer.bias_log_sigma)
            x=F.linear(x, weight_, bias_)
            x = self.activation()(x)
            
            multipliers = m.shape[1] / (m.sum(1) + 1e-6)
            x = torch.mul((x * m).T, multipliers).T

            ####probabilitty of picking Bayesian neural network 
            CDF1=torch.distributions.normal.Normal(layer.weight_mu,torch.exp(layer.weight_log_sigma)).cdf(weight_+1e-5)
            CDF2=torch.distributions.normal.Normal(layer.weight_mu,torch.exp(layer.weight_log_sigma)).cdf(weight_-1e-5)
            p_weights=CDF1-CDF2
            log_p_weights=torch.log(p_weights)

            CDFb1=torch.distributions.normal.Normal(layer.bias_mu,torch.exp(layer.bias_log_sigma)).cdf(bias_+1e-5)
            CDFb2=torch.distributions.normal.Normal(layer.bias_mu,torch.exp(layer.bias_log_sigma)).cdf(bias_-1e-5)
            p_bias=CDFb1-CDFb2
            log_p_bias=torch.log(p_bias)

            ###add P_F_wb to the GFN loss
            batch_GFN_loss+=log_p_weights.sum()+log_p_bias.sum()
            ###minus P_B_wb to the GFN loss
            batch_GFN_loss-=0 #as the weights and biases are sampled in parallel , P_B=1
        
        ####output final prediction
        pred = self.out_layer(x)

        #### negative log likelihood loss
        CEloss = nn.CrossEntropyLoss(reduce=None)(pred, y)

        NLL=Data_set_size*CEloss

        batch_GFN_loss+=NLL
        ####regularitization on weights
        decay_coeff=0.01 #hyperparameters
        all_weights_mag=0
        for layer in self.fc:
            all_weights_mag+=(layer.weight_mu**2).sum()
            all_weights_mag+=torch.exp(layer.weight_log_sigma).sum()

        WeightDecay=decay_coeff*(all_weights_mag)


        batch_GFN_loss+=WeightDecay


        batch_GFN_loss=batch_GFN_loss**2

        return pred,batch_GFN_loss,masks_mu,masks_z

class BaysianMLPClassifierWithMaskGenerator(object):
    def __init__(
            self,
            in_dim=784,
            out_dim=10,
            hidden=None,
            activation=nn.LeakyReLU,
            dropout_rate=0.5,
            mg_type='random',
            lr=1e-3,
            z_lr=1e-1,
            mg_lr=1e-2,
            mg_hidden=None,
            mg_activation=nn.LeakyReLU,
            beta=0.1,
            device='cpu',
    ):
        # classifier
        self.model = BaysianMLPMaskedDropout(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            activation=activation
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


            self.z_mask_generators = construct_mlp_z_mask_generators(
                model=self.model,
                hidden=mg_hidden,
                activation=mg_activation,
            ).to(device)


            self.mu_P_b = construct_mu_backwards_policy(
                model=self.model,
                hidden=mg_hidden,
                activation=mg_activation,
            ).to(device)

            #self.logZ = nn.Parameter(torch.tensor(0.)).to(device)
            self.total_flowestimator=MLP(in_dim=in_dim,out_dim=1,
                                    activation=mg_activation).to(device)



            param_list = [{'params': self.model.parameters(), 'lr': lr},
                        {'params': self.mu_mask_generators.parameters(), 'lr': mg_lr},
                        {'params': self.z_mask_generators.parameters(), 'lr': mg_lr},
                        {'params': self.mu_P_b.parameters(), 'lr': mg_lr},
                        {'params': self.total_flowestimator.parameters(), 'lr': z_lr}]

            self.optimizer = optim.Adam(param_list)
        else:
            raise ValueError('unknown mask generator type {}'.format(mg_type))

    def step(self, x, y,Data_set_size):
        metric = {}
        logits,loss, mu_masks,z_masks = self.model(x,y, 
                                                    self.mu_mask_generators,
                                                    self.z_mask_generators,
                                                    self.mu_P_b,
                                                    self.total_flowestimator,
                                                    Data_set_size)
        # Update model
        self.optimizer.zero_grad()

        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
        metric['loss'] = loss.item()
        metric['acc'] = acc

        loss.sum().backward()
        
        self.optimizer.step()

     
        ###calculated actual droppout rate
        actual_dropout_rate=0


        n_units=0
        n_dropped=0.0
        batch_size=x.shape[0]
        for layer_idx in range(len(self.model.fc)):
            m=mu_masks[layer_idx][-1].unsqueeze(0).repeat(batch_size,1)*z_masks[layer_idx]
            n_units+=m.shape[1]
            n_dropped=(m==0).mean(0).sum()

        for m in masks:
            actual_dropout_rate+=m.sum(1).mean(0)/(m.shape[1]) 
        metric['actual_dropout_rate']=1-(actual_dropout_rate/len(masks)).item()


        return metric


    def test(self, x, y):
        metric = {}
        SampleRepeats=5

        with torch.no_grad():
            logits=[]
            masks=[]
            for _ in range(SampleRepeats):
                logits_, _ ,_,_= self.model(x,y, 
                                            self.mu_mask_generators,
                                            self.z_mask_generators,
                                            self.mu_P_b,
                                            self.total_flowestimator,
                                            1)###y is used for loss caculation but not prediction
                logits.append(logits_.unsqueeze(2))

                masks_=[]
                for layer_idx in range(len(self.model.fc)):
                    m=mu_masks[layer_idx][-1].unsqueeze(0).repeat(batch_size,1)*z_masks[layer_idx]
                    masks_.append(m)

                masks_=torch.cat(masks_,1)
                masks.append(masks_.unsqueeze(2))
            logits=torch.log((torch.exp(torch.cat(logits,2)).mean(2)))
            masks=torch.cat(masks,2)
            
            loss = nn.CrossEntropyLoss()(logits, y)
            acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
            metric['loss'] = loss.item()
            metric['acc'] = acc

        return metric,masks
