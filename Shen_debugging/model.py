import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F


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


class MLPMaskedDropout(nn.Module):

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

    def forward(self, x, mask_generators):
        masks = []
        intermediates=[]
        for layer, mg in zip(self.fc, mask_generators):
            x = self.activation()(layer(x))
            intermediates.append(x.detach())
            # generate mask & dropout
            m = mg(x).detach()
            masks.append(m)
            multipliers = m.shape[1] / (m.sum(1) + 1e-6)
            x = torch.mul((x * m).T, multipliers).T
        x = self.out_layer(x)
        return x, masks,intermediates


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


class MLPMaskGenerator(nn.Module):
    def __init__(self, num_unit, dropout_rate, hidden=None, activation=nn.LeakyReLU):
        super().__init__()
        self.num_unit = torch.tensor(num_unit).type(torch.float32)
        self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)
        self.mlp = MLP(
            in_dim=num_unit,
            out_dim=num_unit,
            hidden=hidden,
            activation=activation,
        )

    def _dist(self, x):
        x = self.mlp(x)
        #x = torch.sigmoid(x)
        if self.train:
            T=0.1
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


def construct_mlp_mask_generators(
        model,
        dropout_rate,
        hidden=None,
        activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()
    for layer in model.fc:
        mask_generators.append(
            MLPMaskGenerator(
                num_unit=layer.weight.shape[0],
                dropout_rate=dropout_rate,
                hidden=hidden,
                activation=activation
            )
        )

    return mask_generators


class MLPClassifierWithMaskGenerator(object):
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
            mg_lr=1e-3,
            mg_hidden=None,
            mg_activation=nn.LeakyReLU,
            beta=0.1,
            device='cpu',
    ):
        # classifier
        self.model = MLPMaskedDropout(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            activation=activation
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # mask generators
        self.mg_type = mg_type
        if mg_type == 'random':
            self.mask_generators = construct_random_mask_generators(
                model=self.model,
                dropout_rate=dropout_rate
            ).to(device)
        elif mg_type == 'gfn':
            # for backward log prob calculation only
            self.rand_mask_generators = construct_random_mask_generators(
                model=self.model,
                dropout_rate=dropout_rate
            ).to(device)
            self.mask_generators = construct_mlp_mask_generators(
                model=self.model,
                dropout_rate=dropout_rate,
                hidden=mg_hidden,
                activation=mg_activation,
            ).to(device)
            #self.logZ = nn.Parameter(torch.tensor(0.)).to(device)
            self.total_flowestimator=MLP(in_dim=in_dim,out_dim=1,
                                    activation=mg_activation).to(device)


            # param_list = [{'params': self.model.parameters(), 'lr': mg_lr},
            #               {'params': self.logZ, 'lr': z_lr}]

            param_list = [{'params': self.model.parameters(), 'lr': mg_lr},
                        {'params': self.total_flowestimator.parameters(), 'lr': z_lr}]

            self.mg_optimizer = optim.Adam(param_list)
        else:
            raise ValueError('unknown mask generator type {}'.format(mg_type))

        # gfn parameters
        self.beta = beta

    def step(self, x, y, x_valid=None, y_valid=None):
        metric = {}
        logits, masks,_ = self.model(x, self.mask_generators)
        # Update model
        self.optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
        metric['loss'] = loss.item()
        metric['acc'] = acc
        loss.backward()
        self.optimizer.step()

        # Update mask generators
        if self.mg_type == 'gfn':
            if x_valid is not None and y_valid is not None:
                metric.update(self._gfn_step(x_valid, y_valid))
            else:
                metric.update(self._gfn_step(x, y))


     
        ###calculated actual droppout rate
        actual_dropout_rate=0
        for m in masks:
            actual_dropout_rate+=m.sum(1).mean(0)/(m.shape[1]) 
        metric['actual_dropout_rate']=1-(actual_dropout_rate/len(masks)).item()


        return metric

    def _gfn_step(self, x, y):
        metric = {}
        logits, masks,intermediates = self.model(x, self.mask_generators)
        with torch.no_grad():
            losses = nn.CrossEntropyLoss(reduce=False)(logits, y)
            log_rewards = - self.beta * losses
        # trajectory balance loss
        log_probs_F = []
        log_probs_B = []
        for interm,m, mg_f, mg_b in zip(intermediates,masks, self.mask_generators, self.rand_mask_generators):
            log_probs_F.append(mg_f.log_prob(interm, m).unsqueeze(1))
            #log_probs_B.append(mg_b.log_prob(m, m).unsqueeze(1))
            log_probs_B.append(torch.zeros(m.shape[0]).unsqueeze(1))
            
        # tb_loss = ((self.logZ - log_rewards
        #             + torch.cat(log_probs_F, dim=1).sum(dim=1)
        #             - torch.cat(log_probs_B, dim=1).sum(dim=1)) ** 2).mean()


        logZ=self.total_flowestimator(x)#this flow is calculated using x_mask, not a bug , to encourage generalization 

        tb_loss = ((logZ - log_rewards
                            + torch.cat(log_probs_F, dim=1).sum(dim=1)
                            - torch.cat(log_probs_B, dim=1).sum(dim=1)) ** 2).mean()

        metric['tb_loss'] = tb_loss.item()
        self.mg_optimizer.zero_grad()
        tb_loss.backward()
        self.mg_optimizer.step()

        return metric

    def test(self, x, y):
        metric = {}
        SampleRepeats=5

        logits=[]
        masks=[]
        for _ in range(SampleRepeats):
            logits_, masks_ ,_= self.model(x, self.mask_generators)
            logits.append(logits_.unsqueeze(2))
            masks_=torch.cat(masks_,1)
            masks.append(masks_.unsqueeze(2))
        logits=torch.log((torch.exp(torch.cat(logits,2)).mean(2)))
        masks=torch.cat(masks,2)
        
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
        metric['loss'] = loss.item()
        metric['acc'] = acc

        return metric,masks
