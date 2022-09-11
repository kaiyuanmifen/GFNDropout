# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch
import numpy as np

class FC(nn.Module):
    def __init__(self, in_size, out_size, input_dim=None, dropout_r=0., use_relu=True, HP=None):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if HP.concretedp:
            self.dropout_regularizer = 2. / HP.data_size
        else:
            self.dropout_regularizer = 1.0
        self.dropout = Dropout_variants(dropout_regularizer=self.dropout_regularizer, dp_type=HP.dp_type,
                                        concretedp=HP.concretedp, learn_prior=HP.learn_prior,
                                        k=HP.dp_k, eta_const=HP.eta_const, cha_factor=32, ARM=HP.ARM,
                                        dropout_dim=2, input_dim=input_dim,
                                        dropout_distribution=HP.dropout_distribution, ctype = HP.ctype)
        # self.dropout = nn.Dropout(dropout_r)
    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)
        #print('fc shape', x.shape)   # 64, 100, 2048 // 64, 14, 512, // 64, 100, 512
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True, HP=None, input_dim=None):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu, HP=HP, input_dim=input_dim)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Dropout_variants(nn.Module):
    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5, dp_type=False, ARM=True, concretedp=True,
                 learn_prior=False, k=1, eta_const=-1.38, cha_factor=1, dropout_dim=-1, input_dim=[1],
                 dropout_distribution='bernoulli', ctype = "Gaussian"):
        super(Dropout_variants, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.cha_factor = cha_factor
        self.dp_type = dp_type
        self.ARM = ARM
        self.concretedp = concretedp
        self.learn_prior = learn_prior
        self.k = k
        self.use_bias = True
        self.dropout_dim = dropout_dim
        self.input_dim = input_dim
        self.forward_mode = True
        self.dropout_distribution = dropout_distribution
        self.ctype = ctype
        out_features = input_dim[dropout_dim]

        if dp_type:
            self.ctdo_linear2 = nn.Linear(out_features, int(out_features/self.cha_factor), bias=self.use_bias)
            self.ctdo_linear3 = nn.Linear(int(out_features/self.cha_factor), out_features, bias=self.use_bias)
            self.ctdo_linear3.weight.data.normal_(0, 1 * np.sqrt(float(self.cha_factor) / out_features))  # TODO: tune
            self.ctdo_linear2.weight.data.normal_(0, 1 * np.sqrt(1.0 / out_features))
            if self.use_bias:
                self.ctdo_linear3.bias.data.fill_(eta_const)
                self.ctdo_linear2.bias.data.fill_(eta_const)
                # self.ctdo_linear3.weight.data.normal_(0, 1/10) #TODO: tune
                # self.ctdo_linear2.weight.data.normal_(0, 1/10)
                # l0_arm is the below
                # self.att_weights1.data.normal_(0, 1 * np.sqrt(1.0 / self.out_channels))
                # self.att_weights2.data.normal_(0, 1 * np.sqrt(float(self.cha_factor) / self.out_channels))


            if learn_prior:
                self.eta = nn.Parameter(torch.Tensor(1))
                self.eta.data.fill_(eta_const)
            else:
                self.eta = (torch.from_numpy(np.ones([1])) * eta_const).type(torch.float32).cuda()
        else:
            if concretedp:
                self.z_phi = nn.Parameter(torch.empty(1).data.fill_(eta_const))
                self.eta = (torch.from_numpy(np.ones([1])) * eta_const).type(torch.float32).cuda()
            else:
                self.z_phi = torch.from_numpy(np.ones([1]) * eta_const).type(torch.float32).cuda()

    def contextual_dropout(self, input):
        # input: batch, x,..., dropout_dim, ..., y
        # output: batch, 1..., dropout_dim, ..., 1
        if self.dp_type:
            # first squeeze
            input_cp = input
            for dim in range(1, len(self.input_dim)):
                if dim != self.dropout_dim:
                    input_cp = torch.mean(input_cp, dim=dim, keepdim=True)
            input_cp = torch.transpose(input_cp, -1, self.dropout_dim)

            # then excite
            z_phi = self.ctdo_linear2(input_cp.data)
            # m = nn.LeakyReLU(0.1) #TODO: tune.
            m = nn.ReLU()  # TODO: tune.
            # m = nn.Tanh()
            z_phi = m(z_phi)
            z_phi = self.ctdo_linear3(z_phi)
            z_phi = torch.transpose(z_phi, -1, self.dropout_dim)
            return z_phi

    def _dropout(self, x, pi):
        eps = 1e-7
        temp = 0.1
        if self.dp_type and self.ctype == "Gaussian":
            mu_prior = torch.ones_like(pi).cuda() 
            mu_post = torch.ones_like(pi).cuda() 
            
            self.u = torch.randn_like(pi)
            
            prior_pi = torch.sigmoid(self.k * self.eta)
            prior_sigma = torch.sqrt(prior_pi/(1-prior_pi))

            sigma = torch.sqrt(pi/(1-pi))

            prior_dist = torch.distributions.normal.Normal(mu_prior, prior_sigma)
            post_dist = torch.distributions.normal.Normal(mu_post, sigma)

            z = mu_post + sigma * self.u


            self.post_nll_true = (-post_dist.log_prob(z))
            self.prior_nll_true = (-prior_dist.log_prob(z))

            return torch.mul(x, z)
        elif self.dp_type and self.ARM:
            if self.forward_mode:
                # if len(x.shape) < 4:
                #     self.u = torch.rand_like(x)
                # else:
                self.u = torch.rand_like(pi)
                drop_prob = (self.u < pi).type_as(pi)
            else:
                drop_prob = (self.u < 1 - pi).type_as(pi)
        else:
            # change dimension
            if self.concretedp:
                unif_noise = torch.rand_like(pi)
                drop_prob = (torch.log(pi + eps)
                             - torch.log(1 - pi + eps)
                             + torch.log(unif_noise + eps)
                             - torch.log(1 - unif_noise + eps))
                drop_prob = torch.sigmoid(drop_prob / temp)
            else:
                if self.dropout_distribution == 'bernoulli':
                    unif_noise = torch.rand_like(pi)
                    drop_prob = (unif_noise < pi).type_as(pi)
                elif self.dropout_distribution == 'gaussian':
                    sd = torch.sqrt(pi/(1-pi))
                    m = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([sd]))
                    noise = m.sample(sample_shape=pi.shape).type_as(pi)
                    return torch.mul(x, noise)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - pi

        if self.dp_type or self.concretedp:
            self.post_nll_true = - (drop_prob * torch.log(pi + eps) + (1 - drop_prob) * torch.log(1 - pi + eps))
            self.post_nll_true = self.post_nll_true
            prior_pi = torch.sigmoid(self.k * self.eta).type_as(self.post_nll_true)
            self.prior_nll_true = - (drop_prob * torch.log(prior_pi + eps) + (1 - drop_prob) * torch.log(1 - prior_pi + eps))
            self.prior_nll_true = self.prior_nll_true
        x = torch.mul(x, random_tensor)
        x /= retain_prob #TODO: think
        return x

    def forward(self, x):
        input_dimensionality = x[0].numel()
        if self.dp_type:
            self.z_phi = self.contextual_dropout(x)
        pi = torch.sigmoid(self.k * self.z_phi)

        if np.random.uniform() > 0.9999:
            print('mean', torch.mean(pi))
            print('std', torch.std(pi))
            if self.dp_type or self.concretedp:
                print('eta', torch.mean(torch.sigmoid(self.eta * self.k)))

        if self.training:
            x = self._dropout(x, pi)
            if self.dp_type:
                if self.ctype != "Gaussian" and self.ARM:
                    regularization = 0
                else:
                    regularization = ((self.post_nll_true - self.prior_nll_true) * self.dropout_regularizer).mean()
                    regularization.backward(retain_graph=True)
            else:
                if self.concretedp:
                    regularization = ((self.post_nll_true - self.prior_nll_true) * self.dropout_regularizer).mean()
                    regularization.backward(retain_graph=True)
                else:
                    regularization = 0
        # else:
        #     x = x * (1 - pi)
        return x


    def update_phi_gradient(self, f1, f2):
        #TODO: define u, f1 and f2.
        k = self.k
        z_phi_shape = len(self.z_phi.shape)
        for _ in range(z_phi_shape-1):
            f1 = f1.unsqueeze(1)
            f2 = f2.unsqueeze(1)
        # if len(self.u.shape) > 2:
        #     for dim in range(1, len(self.input_dim)):
        #         if dim != self.dropout_dim:
        #             self.u = torch.mean(self.u, dim=dim, keepdim=True)
        e = k * ((f1 - f2) * (self.u - .5))
        self.z_phi.backward(e, retain_graph=True)