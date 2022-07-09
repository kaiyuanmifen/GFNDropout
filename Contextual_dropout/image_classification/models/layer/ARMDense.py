import math

import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from six.moves import cPickle

epsilon = 1e-10


class ARMDense(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_decay=1e-4, lamba=0.001, droprate_init=.5,
                 local_rep=False,opt=None, **kwargs):
        super(ARMDense, self).__init__()

        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.opt=opt
        self.in_features = in_features
        self.out_features = out_features
        self.weight_decay = weight_decay
        self.lamba = lamba
        self.weights = nn.Parameter(torch.Tensor(in_features, out_features, ))
        self.droprate_init = droprate_init
        self.sum_pi = 0.0
        self.mask_threshold = 0.0
        # flag
        if self.opt.var_dropout:
            self.log_sigma = nn.Parameter(torch.Tensor(in_features, out_features, ))
            self.clip_alpha = 10
        elif self.opt.dptype:
            self.ctdo_linear2 = nn.Linear(in_features, int(in_features / self.opt.cha_factor), bias=True)
            self.ctdo_linear3 = nn.Linear(int(in_features / self.opt.cha_factor), in_features, bias=True)
            self.ctdo_linear3.weight.data.normal_(0, 1 * np.sqrt(float(self.opt.cha_factor) / in_features))  # TODO: tune
            self.ctdo_linear2.weight.data.normal_(0, 1 * np.sqrt(1.0 / in_features))
            self.ctdo_linear3.bias.data.normal_(
                (math.log(1 - self.droprate_init) - math.log(self.droprate_init)) / self.opt.k,
                1e-2 / self.opt.k)
            self.ctdo_linear2.bias.data.normal_(
                (math.log(1 - self.droprate_init) - math.log(self.droprate_init)) / self.opt.k,
                1e-2 / self.opt.k)
            if self.opt.learn_prior:
                if self.opt.learn_prior_scalar:
                    # for scalar we might need smaller learning rate
                    self.learn_scaler = nn.Parameter(
                        torch.Tensor(1))  # .to(self.device) if self.opt.use_gpu else self.learn_scaler.type_as(self.weights)
                    # self.eta = self.learn_scaler
                    # self.eta = torch.from_numpy(np.ones(in_features)).type_as(self.weights) * self.learn_scaler
                    # a = torch.from_numpy(np.ones(in_features)).type_as(self.weights).to(self.device) if self.opt.use_gpu else a.type_as(self.weights)
                    # self.eta= a * self.learn_scaler
                    self.eta = self.learn_scaler
                    # self.eta = self.eta.to(self.device) if self.opt.use_gpu else self.eta
                else:
                    self.eta = nn.Parameter(torch.Tensor(in_features))
            else:
                self.eta = torch.from_numpy(np.ones(in_features)) * self.opt.eta
                self.eta = self.eta.type_as(self.weights).to(self.device) if self.opt.use_gpu else self.eta.type_as(self.weights)
        else:
            if self.opt.concretedp:
                if self.opt.concretedp_cha:
                    self.z_phi = nn.Parameter(torch.Tensor(in_features))
                    self.eta = torch.from_numpy(np.ones(in_features)) * self.opt.eta
                    self.eta = self.eta.type_as(self.weights).to(self.device) if self.opt.use_gpu else self.eta.type_as(self.weights)
                else:
                    self.z_phi = nn.Parameter(torch.Tensor(1))
                    self.eta = torch.from_numpy(np.ones(in_features)) * self.opt.eta
                    self.eta = self.eta.type_as(self.weights).to(self.device) if self.opt.use_gpu else self.eta.type_as(self.weights)
            else:
                if self.opt.fixdistrdp:
                    self.z_phi = torch.from_numpy(138.6 * np.ones(in_features)).type_as(self.weights)
                    self.z_phi = self.z_phi.to(self.device) if self.opt.use_gpu else self.z_phi
                else:
                    self.z_phi = nn.Parameter(torch.Tensor(in_features))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.floatTensor = torch.FloatTensor if not self.opt.use_gpu else torch.cuda.FloatTensor

        self.u = None
        self.forward_mode = True
        self.local_rep = local_rep
        self.concrete_temp = self.opt.concrete_temp
        self.reset_parameters()
        print(self)






    def contextual_dropoutdense(self, input):
        if self.opt.dptype:
            if self.opt.se:
                #def contextual_dropoutdense(self, input):
                    # input shape: batch, channel, weight, height
                    # output shape: batch, 1, weight, height
                    # print(input.shape)
                    z_phi = self.ctdo_linear2(input)
                    # z_phi = self.ctdo_linear2(input)
                    # z_phi = nn.Tanh()(z_phi)

                    m = nn.LeakyReLU(0.1)
                    z_phi = m(z_phi)

                    # z_phi = F.relu(z_phi)
                    z_phi = self.ctdo_linear3(z_phi)
                    return z_phi
            else:
                #def contextual_dropoutdense(self, input):
                    # input shape: batch, channel, weight, height
                    # output shape: batch, 1, weight, height
                    # print(input.shape)
                    # z_phi = self.ctdo_linear2(input.data)
                    z_phi = self.ctdo_linear2(input.data)
                    # z_phi = nn.Tanh()(z_phi)

                    m = nn.LeakyReLU(0.1)
                    z_phi = m(z_phi)

                    # z_phi = F.relu(z_phi)
                    z_phi = self.ctdo_linear3(z_phi)
                    return z_phi

    def sample_z(self, batch_size=None,input=None):
        ###sample z function
        
        # flag
        if self.opt.dptype:
            if self.opt.ctype == "Gaussian":
                #def sample_z(self, batch_size, input):
                    # input: batch, channel, weight, height
                    # output: batch, 1, weight, height
                    # self.z_phi = self.some_func(input)
                    self.z_phi = self.contextual_dropoutdense(input)
                    
                    mu_prior = torch.ones((batch_size, self.in_features)).to(self.device) 
                    mu_post = torch.ones((batch_size, self.in_features)).to(self.device) 
                    
                    prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0))
                    prior_sigma = torch.sqrt((1-prior_pi)/prior_pi)
                    #prior_sigma = torch.exp(self.opt.temp * self.eta * torch.ones((batch_size, self.in_features)).to(self.device)) 

                    self.u = self.floatTensor(batch_size, self.in_features).normal_(0, 1)
                    u = self.u
                    #sigma = torch.exp(self.opt.temp * self.z_phi)
                    pi = torch.sigmoid(self.opt.k * self.z_phi)
                    sigma = torch.sqrt((1-pi)/pi)

                    prior_dist = torch.distributions.normal.Normal(mu_prior, prior_sigma)
                    post_dist = torch.distributions.normal.Normal(mu_post, sigma)
                    if self.training:
                        ##Reparameterization trick
                        z = mu_post + sigma * u
                        ##For later use when computing KL
                        self.train_z = z
                    else:
                        if self.opt.test_sample_mode == 'greedy':
                            z = torch.ones((batch_size, self.in_features)).to(self.device) 
                        else:
                            z = mu_post + sigma * u
                        self.test_z = z
                    self.prior_nll_true = (-prior_dist.log_prob(z)).mean(1)
                    self.post_nll_true = (-post_dist.log_prob(z)).mean(1)
                    return z
            else:
                #def sample_z(self, batch_size, input):
                    # input: batch, channel, weight, height
                    # output: batch, 1, weight, height
                    # self.z_phi = self.some_func(input)
                    self.z_phi = self.contextual_dropoutdense(input)
                    if self.opt.hardsigmoid:
                        pi = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1)
                    else:
                        pi = torch.sigmoid(self.opt.k * self.z_phi)
                    # updating
                    if np.random.uniform() > 0.9999:
                        print('mean', torch.mean(pi))
                        print('std', torch.std(pi))
                        print('eta', torch.mean(torch.sigmoid(self.eta * self.opt.k)))
                    self.pi = pi
                    if self.forward_mode:
                        z = self.floatTensor(batch_size, self.in_features).zero_()
                        if self.training:
                            if self.opt.add_pi:
                                self.sum_pi = self.sum_pi + self.pi.mean(0)  # * self.weights.mean(1)
                            if self.local_rep:
                                self.u = self.floatTensor(self.in_features).uniform_(0, 1).expand(batch_size,
                                                                                                  self.in_features)
                            else:
                                self.u = self.floatTensor(batch_size, self.in_features).uniform_(0, 1)
                            if self.opt.optim_method:
                                pi = torch.sigmoid(self.opt.k * self.z_phi)
                                eps = 1e-7
                                temp = self.concrete_temp
                                u = self.u
                                z = (torch.log(pi + eps) - torch.log(1 - pi + eps) + torch.log(u + eps) - torch.log(
                                    1 - u + eps))
                                z = torch.sigmoid(z / temp)
                            else:
                                z[self.u < pi] = 1
                                if self.opt.use_t_in_training:
                                    z[pi < self.opt.t] = 0
                            if self.opt.finetune:
                                z[self.test_z[0:1, :].repeat(batch_size, 1) == 0] = 0
                            self.train_z = z
                            self.post_nll_true = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                            self.post_nll_true = self.post_nll_true.mean(1)
                            prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0))
                            self.prior_nll_true = - (
                                        z * torch.log(prior_pi + epsilon) + (1 - z) * torch.log(1 - prior_pi + epsilon))
                            self.prior_nll_true = self.prior_nll_true.mean(1)

                            self.post_nll_true_rb = - (
                                        pi * torch.log(pi + epsilon) + (1 - pi) * torch.log(1 - pi + epsilon))
                            self.post_nll_true_rb = self.post_nll_true_rb.mean(1)
                            self.prior_nll_true_rb = - (
                                        pi * torch.log(prior_pi + epsilon) + (1 - pi) * torch.log(1 - prior_pi + epsilon))
                            self.prior_nll_true_rb = self.prior_nll_true_rb.mean(1)
                        else:
                            if self.opt.test_sample_mode == 'greedy':
                                z[self.z_phi > 0] = 1
                                # self.test_z = z
                                if self.opt.use_t_in_testing:
                                    z = pi
                                    if self.opt.use_uniform_mask:
                                        if self.opt.finetune:
                                            z[self.test_z[0:1, :].repeat(batch_size, 1) == 0] = 0
                                        else:
                                            if self.opt.mask_type == 'prior':
                                                mask = (torch.sigmoid(self.opt.k * self.eta) < self.opt.t).unsqueeze(0).repeat(
                                                    batch_size, 1)
                                            elif self.opt.mask_type == 'pi_sum':
                                                mask = (self.sum_pi < self.mask_threshold).unsqueeze(0).repeat(batch_size,
                                                                                                               1)
                                            z[mask] = 0
                                            self.test_z = 1 - mask.type_as(z)
                                    else:
                                        z[z < self.opt.t] = 0
                                        self.test_z = z
                            else:
                                self.u = self.floatTensor(batch_size, self.in_features).uniform_(0, 1)
                                z[self.u < pi] = 1
                                # self.test_z = z
                                if self.opt.use_uniform_mask:
                                    if self.opt.finetune:
                                        z[self.test_z[0:1, :].repeat(batch_size, 1) == 0] = 0
                                    else:
                                        if self.opt.mask_type == 'prior':
                                            mask = (torch.sigmoid(self.opt.k * self.eta) < self.opt.t).unsqueeze(0).repeat(batch_size,
                                                                                                                 1)
                                        elif self.opt.mask_type == 'pi_sum':
                                            mask = (self.sum_pi < self.mask_threshold).unsqueeze(0).repeat(batch_size, 1)
                                        z[mask] = 0
                                        self.test_z = 1 - mask.type_as(z)
                            self.post_nll_true = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                            self.post_nll_true = self.post_nll_true.mean(1)
                            prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0))
                            self.prior_nll_true = - (
                                    z * torch.log(prior_pi + epsilon) + (1 - z) * torch.log(1 - prior_pi + epsilon))
                            self.prior_nll_true = self.prior_nll_true.mean(1)
                    else:
                        pi2 = 1 - pi
                        if self.u is None:
                            raise Exception('Forward pass first')
                        z = self.floatTensor(self.u.size()).zero_()
                        z[self.u > pi2.expand(batch_size, self.in_features)] = 1
                        if self.opt.finetune:
                            z[self.test_z[0:1, :].repeat(batch_size, 1) == 0] = 0
                        z_true = self.floatTensor(self.u.size()).zero_()
                        z_true[self.u < pi] = 1
                        if self.opt.use_t_in_training:
                            z[pi.expand(batch_size, self.in_features) < self.opt.t] = 0
                        self.sarm_mask = pi.new_ones(batch_size, self.in_features)
                        self.sarm_mask[z == z_true] = 0.0
                        self.new_pseudo = self.sarm_mask.sum() != 0
                        self.post_nll_sudo = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                        self.post_nll_sudo = self.post_nll_sudo.mean(1)
                        prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0))
                        self.prior_nll_sudo = - (
                                    z * torch.log(prior_pi + epsilon) + (1 - z) * torch.log(1 - prior_pi + epsilon))
                        self.prior_nll_sudo = self.prior_nll_sudo.mean(1)

                        self.post_nll_sudo_rb = - (pi * torch.log(pi + epsilon) + (1 - pi) * torch.log(1 - pi + epsilon))
                        self.post_nll_sudo_rb = self.post_nll_sudo_rb.mean(1)
                        self.prior_nll_sudo_rb = - (
                                pi * torch.log(prior_pi + epsilon) + (1 - pi) * torch.log(1 - prior_pi + epsilon))
                        self.prior_nll_sudo_rb = self.prior_nll_sudo_rb.mean(1)

                    return z
        else:
            #def sample_z(self, batch_size):
                if self.opt.hardsigmoid:
                    pi = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1).detach()
                else:
                    pi = torch.sigmoid(self.opt.k * self.z_phi).detach()
                self.pi = pi
                if np.random.uniform() > 0.9999:
                    print('mean', torch.mean(pi))
                    print('std', torch.std(pi))
                    # if not self.opt.fixdistrdp:
                    #     print('eta', torch.mean(torch.sigmoid(self.eta * self.opt.k)))
                if self.forward_mode:
                    z = self.floatTensor(batch_size, self.in_features).zero_()
                    if self.training:
                        if self.local_rep:
                            self.u = self.floatTensor(self.in_features).uniform_(0, 1).expand(batch_size, self.in_features)
                        else:
                            self.u = self.floatTensor(batch_size, self.in_features).uniform_(0, 1)
                        if self.opt.gumbelconcrete:
                            pi = torch.sigmoid(self.opt.k * self.z_phi)
                            eps = 1e-7
                            temp = self.concrete_temp
                            u = self.u
                            z = (torch.log(pi + eps) - torch.log(1 - pi + eps) + torch.log(u + eps) - torch.log(
                                1 - u + eps))
                            z = torch.sigmoid(z / temp)
                        else:
                            z[self.u < pi.expand(batch_size, self.in_features)] = 1
                        if self.opt.use_t_in_training:
                            z[pi.expand(batch_size, self.in_features) < self.opt.t] = 0

                        if self.opt.finetune:
                            z[self.test_z[0:1, :].repeat(batch_size, 1) == 0] = 0
                            # self.test_z = z
                        self.train_z = z
                        if self.opt.concretedp:
                            pi = torch.sigmoid(self.opt.k * self.z_phi)
                            self.post_nll_true = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                            self.post_nll_true = self.post_nll_true.mean(1)

                            prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0))
                            self.prior_nll_true = - (
                                        z * torch.log(prior_pi + epsilon) + (1 - z) * torch.log(1 - prior_pi + epsilon))
                            self.prior_nll_true = self.prior_nll_true.mean(1)

                    else:
                        if self.opt.test_sample_mode == 'greedy':
                            z[self.z_phi.expand(batch_size, self.in_features) > 0] = 1

                            if self.opt.use_t_in_testing:
                                z = pi.expand(batch_size, self.in_features)
                                if self.opt.finetune:
                                    z[self.test_z[0:1, :].repeat(batch_size, 1) == 0] = 0
                                else:
                                    z[z < self.opt.t] = 0
                                    self.test_z = z
                        else:
                            self.u = self.floatTensor(batch_size, self.in_features).uniform_(0, 1)
                            z[self.u < pi.expand(batch_size, self.in_features)] = 1
                            if self.opt.finetune:
                                z[self.test_z[0:1, :].repeat(batch_size, 1) == 0] = 0
                            else:
                                z[pi.expand(batch_size, self.in_features) < self.opt.t] = 0
                            # self.test_z = self.opt.t < pi.expand(batch_size, self.in_features)
                        if self.opt.concretedp:
                            self.post_nll_true = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                            self.post_nll_true = self.post_nll_true.mean(1)
                            prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0))
                            self.prior_nll_true = - (
                                    z * torch.log(prior_pi + epsilon) + (1 - z) * torch.log(1 - prior_pi + epsilon))
                            self.prior_nll_true = self.prior_nll_true.mean(1)
                else:
                    pi2 = 1 - pi
                    if self.u is None:
                        raise Exception('Forward pass first')
                    z = self.floatTensor(self.u.size()).zero_()
                    z[self.u > pi2.expand(batch_size, self.in_features)] = 1
                    if self.opt.finetune:
                        z[self.test_z[0:1, :].repeat(batch_size, 1) == 0] = 0
                    if self.opt.use_t_in_training:
                        z[pi.expand(batch_size, self.in_features) < self.opt.t] = 0
                    if self.opt.concretedp:
                        pi = torch.sigmoid(self.opt.k * self.z_phi)
                        self.post_nll_sudo = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                        self.post_nll_sudo = self.post_nll_sudo.mean(1)
                        prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0))
                        self.prior_nll_sudo = - (
                                z * torch.log(prior_pi + epsilon) + (1 - z) * torch.log(1 - prior_pi + epsilon))
                        self.prior_nll_sudo = self.prior_nll_sudo.mean(1)
                return z



    def reset_parameters(self):
        # flag
        if self.opt.var_dropout:
            nn.init.kaiming_normal_(self.weights, mode='fan_out')
            self.log_sigma.data.fill_(-5)        
            if self.use_bias:
                self.bias.data.fill_(0)
        elif self.opt.dptype:
            nn.init.kaiming_normal_(self.weights, mode='fan_out')
            if self.opt.learn_prior:
                # if True:
                self.eta.data.normal_((math.log(1 - self.droprate_init) - math.log(self.droprate_init)) / self.opt.k,
                                      1e-2 / self.opt.k)
            # self.z_phi.data.normal_((math.log(1 - self.droprate_init) - math.log(self.droprate_init)) / self.opt.k, 1e-2 / self.opt.k)
            if self.use_bias:
                self.bias.data.fill_(0)
        else:
            nn.init.kaiming_normal_(self.weights, mode='fan_out')
            if not self.opt.fixdistrdp:
                self.z_phi.data.normal_((math.log(1 - self.droprate_init) - math.log(self.droprate_init)) / self.opt.k,
                                        1e-2 / self.opt.k)
            if self.use_bias:
                self.bias.data.fill_(0)

    def update_phi_gradient(self, f1, f2):
        # only deal with first part of gradient
        # regularization part will be handled by pytorch
        # flag
        if self.opt.dptype:
            k = self.opt.k
            # input f1 should be 128
            # print('udenseshape')
            # print(self.u.shape)
            f1 = f1.unsqueeze(1)
            f2 = f2.unsqueeze(1)
            if self.opt.ar:
                e = k * (f2 * (1 - 2 * self.u))
            else:
                if self.opt.sparse_arm:
                    e = k * ((f1 - f2) * (self.u - .5)) * self.sarm_mask
                else:
                    e = k * ((f1 - f2) * (self.u - .5))
            if self.opt.finetune:
                e[self.test_z[0:1, :].repeat(self.u.size(0), 1) == 0] = 0
            self.z_phi.backward(e * self.opt.encoder_lr_factor, retain_graph=True)
        else:
            if self.opt.concretedp:
                k = self.opt.k
                f1 = f1.unsqueeze(1)
                f2 = f2.unsqueeze(1)
                if self.opt.ar:
                    e = k * (f2 * (1 - 2 * self.u)).mean(dim=0)
                else:
                    e = k * ((f1 - f2) * (self.u - .5)).mean(dim=0)
                # print('test', self.z_phi.shape, e.mean(dim=0).shape)
                e = e.mean(dim=0, keepdim=True)
                self.z_phi.grad = e
            else:
                k = self.opt.k
                f1 = f1.unsqueeze(1)
                f2 = f2.unsqueeze(1)
                if self.opt.ar:
                    e = k * (f2 * (1 - 2 * self.u)).mean(dim=0)
                else:
                    e = k * ((f1 - f2) * (self.u - .5)).mean(dim=0)
                if self.opt.finetune:
                    e[self.test_z[0, :] == 0] = 0
                # print('test', self.z_phi.shape, e.mean(dim=0).shape)
                self.z_phi.grad = e

    def regularization(self):
        if self.opt.finetune:
            return 0.0
        # flag
        if self.opt.var_dropout:
            return 0.0
        elif self.opt.dptype:
            # if self.opt.hardsigmoid:
            #     pi = F.hardtanh(self.opt.k * self.eta / 7. + .5, 0, 1)
            # else:
            #     pi = torch.sigmoid(self.opt.k * self.eta)
            # l0 = self.lamba * pi.sum() * self.out_features
            # logpw_col = torch.sum(.5 * self.weight_decay * self.weights.pow(2), 1)
            # # print(pi.shape, logpw_col.shape)
            # logpw = torch.sum(pi * logpw_col.unsqueeze(0))
            # logpb = 0 if not self.use_bias else torch.sum(.5 * self.weight_decay * self.bias.pow(2))
            # l2 = logpw + logpb
            # return l0 + l2
            return 0.0
        else:
            ''' similar with L0 paper'''
            # if self.opt.hardsigmoid:
            #     pi = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1)
            # else:
            #     pi = torch.sigmoid(self.opt.k * self.z_phi)

            # l0 = self.lamba * pi.sum() * self.out_features
            # logpw_col = torch.sum(.5 * self.weight_decay * self.weights.pow(2), 1)
            # logpw = torch.sum(pi * logpw_col)
            # logpb = 0 if not self.use_bias else torch.sum(.5 * self.weight_decay * self.bias.pow(2))
            # l2 = logpw + logpb
            # return l0 + l2
            return 0.0

    def count_expected_flops_and_l0(self):
        '''Measures the expected floating point operations (FLOPs) and the expected L0 norm
        dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        + the bias addition for each neuron
        total_flops = (2 * in_features - 1) * out_features + out_features'''

        ##Don't know about this
        if self.opt.var_dropout:
            return 0, 0
        if self.opt.hardsigmoid:
            ppos = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1).sum()
        else:
            ppos = torch.sigmoid(self.opt.k * self.z_phi).sum()
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features

        if not self.opt.ar:
            expected_flops *= 2
            expected_l0 *= 2
        return expected_flops.data, expected_l0.data

    # def some_func(self, input):
    #     input = F.relu(self.fc1(input))
    #     input = self.fc2(input)
    #     return F.log_softmax(input, dim=1)


    def matmul_train(self, input, eps=1e-8):
        mu_activation = torch.mm(input, self.weights)
        if self.use_bias:
            mu_activation += self.bias
        if self.clip_alpha is not None:
            log_alpha = torch.clamp(2.0*self.log_sigma - torch.log(self.weights**2 + eps), -self.clip_alpha, self.clip_alpha)
            log_sigma2 = log_alpha + torch.log(self.weights**2 + eps)
        std_activation = torch.sqrt(torch.mm(input**2, torch.exp(log_sigma2)) + eps)
        return mu_activation + std_activation * self.floatTensor(std_activation.size()).normal_(0, 1)

    def matmul_eval(self, input,  threshold = 3.0, eps=1e-8):
        log_alpha = 2.0*self.log_sigma - torch.log(self.weights**2 + eps)
        masked_weights = (log_alpha <= threshold).float() * self.weights
        self.sparsity_ratio = (log_alpha > threshold).float().mean()
        activation = torch.mm(input, masked_weights)
        if self.use_bias:
            activation += self.bias
        if self.opt.test_sample_mode == 'greedy':
            return activation
        else:
            mu_activation = torch.mm(input, masked_weights)
            if self.use_bias:
                mu_activation += self.bias
            if self.clip_alpha is not None:
                log_alpha = torch.clamp(2.0*self.log_sigma - torch.log(masked_weights**2 + eps), -self.clip_alpha, self.clip_alpha)
                log_sigma2 = log_alpha + torch.log(masked_weights**2 + eps)
            std_activation = torch.sqrt(torch.mm(input**2, torch.exp(log_sigma2)) + eps)
            return mu_activation + std_activation * self.floatTensor(std_activation.size()).normal_(0, 1)

    def get_kl(self, eps=1e-8):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        c = -k1
        log_alpha = 2.0*self.log_sigma - torch.log(self.weights**2 + eps)
        term_1 = k1 * torch.sigmoid(k2 + k3 * log_alpha)
        term_2 = -0.5 * torch.log1p(torch.exp(-log_alpha))
        neg_dkl = term_1 + term_2 + c
        return -torch.sum(neg_dkl)

    def variational_dropout(self, input):
        if self.training:
            return self.matmul_train(input)
        return self.matmul_eval(input)

    def sample_gauss(self, batch_size):
        if self.opt.hardsigmoid:
            pi = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1).detach()
        else:
            pi = torch.sigmoid(self.opt.k * self.z_phi).detach()
        self.pi = pi
        z = self.floatTensor(batch_size, self.in_features).zero_()
        if self.local_rep:
            self.u = self.floatTensor(self.in_features).normal_(1, self.opt.sd).expand(batch_size, self.in_features)
        else:
            self.u = self.floatTensor(batch_size, self.in_features).normal_(1, self.opt.sd)
        z = self.u
        self.train_z = z
        self.test_z = z
        return z

    def forward(self, input):
        """ forward for fc """
        # updating
        # flag
        if self.opt.var_dropout:
            output = self.variational_dropout(input)
            return output
        elif self.opt.dptype:
            if self.opt.se:
                self.z_phi = self.contextual_dropoutdense(input)
                if self.opt.hardsigmoid:
                    self.pi = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1)
                else:
                    self.pi = torch.sigmoid(self.opt.k * self.z_phi)
                xin = input.mul(self.pi)
                self.train_z = self.pi
                self.test_z = self.pi
            else:
                xin = input.mul(self.sample_z(batch_size=input.size(0), input=input))
            output = xin.mm(self.weights)
            if self.use_bias:
                output += self.bias
            return output
        else:
            """ forward for fc """
            if self.opt.dropout_distribution == 'bernoulli':
                xin = input.mul(self.sample_z(batch_size=input.size(0)))
                output = xin.mm(self.weights)
                if self.use_bias:
                    output += self.bias
            elif self.opt.dropout_distribution == 'gaussian':
                if self.training or self.opt.test_sample_mode != 'greedy':
                    xin = input.mul(self.sample_gauss(input.size(0)))
                else:
                    xin = input
                output = xin.mm(self.weights)
                if self.use_bias:
                    output += self.bias
            return output

    def masked_weight(self):
        return self.weights * self.test_z[0].reshape(self.in_features, 1)

    def activated_neurons(self):
        return (self.test_z > 0).sum() / self.test_z.size(0)

    def expected_activated_neurons(self):
        return (self.train_z > 0).sum() / self.train_z.size(0)

    def __repr__(self):
        s = ('{name}({in_features} -> {out_features},'
             'lamba={lamba}, weight_decay={weight_decay}, ')
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

