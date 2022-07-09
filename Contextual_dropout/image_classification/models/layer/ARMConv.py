import math

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.utils import _pair as pair
import torch.nn.functional as F

import numpy as np


import os
from six.moves import cPickle

epsilon = 1e-8

class ArmConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 weight_decay=1.e-4,
                 lamba=0.1 / 6e5, droprate_init=.5, local_rep=False,opt=None, **kwargs):
        super(ArmConv2d, self).__init__()



        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.opt=opt
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.weight_decay = weight_decay
        self.lamba = lamba
        self.floatTensor = torch.FloatTensor if not self.opt.use_gpu else torch.cuda.FloatTensor
        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        self.weights = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        #flag
        if self.opt.var_dropout:
            self.log_sigma = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
            self.clip_alpha = 0
        elif self.opt.dptype:
            if self.opt.dpcha:
                self.att_weights1 = Parameter(torch.Tensor(out_channels, int(out_channels/self.opt.cha_factor)))
                self.att_weights2 = Parameter(torch.Tensor(int(out_channels/self.opt.cha_factor), out_channels))

                if self.opt.learn_prior:
                    if self.opt.learn_prior_scalar:
                        # self.eta = Parameter(torch.Tensor(1))
                        self.learn_scaler = Parameter(torch.Tensor(1))
                        self.eta = self.learn_scaler

                        # self.learn_scaler = Parameter(torch.Tensor(1)).to(self.device)
                        # self.a = torch.from_numpy(np.ones(out_channels) * 0.01).type_as(self.weights).to(self.device)
                        # self.eta = self.a * self.learn_scaler
                        # self.learn_scaler = self.learn_scaler * self.opt.shrink
                        # self.eta = self.learn_scaler
                        # self.a = torch.from_numpy(np.ones(out_channels)).type_as(self.weights).to(self.device)
                        # self.eta = self.a * self.learn_scaler
                        # self.eta = self.learn_scaler


                    else:
                        self.eta = Parameter(torch.Tensor(out_channels))
                else:
                    self.eta = torch.from_numpy(np.ones(out_channels)) * self.opt.eta
                    self.eta = self.eta.type_as(self.weights).to(self.device) if self.opt.use_gpu else self.eta.type_as(self.weights)
                self.att_bias = Parameter(torch.Tensor(out_channels))
            else:
                self.att_weights = Parameter(torch.Tensor(out_channels, 1))
                self.att_bias = Parameter(torch.Tensor(1))
                #self.ctdo_linear1 = nn.Linear(out_channels, 1)
        else:
            if self.opt.concretedp:
                if self.opt.concretedp_cha:
                    self.z_phi = Parameter(torch.Tensor(out_channels))
                    self.eta = torch.from_numpy(np.ones(out_channels)) * self.opt.eta
                    self.eta = self.eta.type_as(self.weights).to(self.device) if self.opt.use_gpu else self.eta.type_as(self.weights)
                else:
                    self.z_phi = nn.Parameter(torch.Tensor(1))
                    #self.eta = torch.from_numpy(np.ones(1)) * self.opt.eta
                    self.eta = torch.from_numpy(np.ones(out_channels)) * self.opt.eta
                    self.eta = self.eta.type_as(self.weights).to(self.device) if self.opt.use_gpu else self.eta.type_as(self.weights)
            else:
                if self.opt.fixdistrdp:
                    self.z_phi = torch.from_numpy(138.6 * np.ones(out_channels))
                    self.z_phi=self.z_phi.type_as(self.weights).to(self.device) if self.opt.use_gpu else self.z_phi.type_as(self.weights)
                    self.z_phi = self.z_phi.to(self.device) if self.opt.use_gpu else self.z_phi
                else:
                    self.z_phi = Parameter(torch.Tensor(out_channels))
        self.cha_factor = self.opt.cha_factor
        self.dim_z = out_channels
        self.input_shape = None
        self.u = torch.Tensor(self.dim_z).uniform_(0, 1)
        self.droprate_init = droprate_init
        self.forward_mode = True
        self.local_rep = local_rep
        self.concrete_temp = self.opt.concrete_temp
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        #flag
        if self.opt.var_dropout:
            nn.init.kaiming_normal_(self.weights, mode='fan_out')
            self.log_sigma.data.fill_(-5)        
            if self.use_bias:
                self.bias.data.fill_(0)
        elif self.opt.dptype:
            nn.init.kaiming_normal_(self.weights, mode='fan_out')
            #self.z_phi.data.normal_((math.log(1 - self.droprate_init) - math.log(self.droprate_init)) / self.opt.k, 1e-2 / self.opt.k)
            if self.use_bias:
                self.bias.data.fill_(0)
            self.att_bias.data.fill_((math.log(1 - self.droprate_init) - math.log(self.droprate_init)) / self.opt.k)
            if self.opt.dpcha:
                self.att_weights1.data.normal_(0, 1*np.sqrt(1.0 / self.out_channels)) # he_init
                self.att_weights2.data.normal_(0, 1*np.sqrt(float(self.cha_factor) / self.out_channels))
                if self.opt.learn_prior:
                    self.eta.data.normal_((math.log(1 - self.droprate_init) - math.log(self.droprate_init)) / self.opt.k, 1e-2 / self.opt.k)
                    print('reset', torch.mean(torch.sigmoid(self.eta * self.opt.k)))
            else:
                self.att_weights.data.normal_(0, 1.0 / self.out_channels)
        else:
            nn.init.kaiming_normal_(self.weights, mode='fan_out')
            if not self.opt.fixdistrdp:
                self.z_phi.data.normal_((math.log(1 - self.droprate_init) - math.log(self.droprate_init)) / self.opt.k,
                                    1e-2 / self.opt.k)
            if self.use_bias:
                self.bias.data.fill_(0)


    def ctdo_linear1(self, input):
        if self.opt.dpcha:

            #output = nn.Tanh()(torch.matmul(input, self.att_weights1))
            m = nn.LeakyReLU(0.1)
            output = m(torch.matmul(input, self.att_weights1))
            #output = F.relu(torch.matmul(input, self.att_weights1))
            #print('lol', torch.abs(input).mean(), F.relu(torch.matmul(output, self.att_weights2)).mean(), torch.abs(self.att_weights1).mean(), torch.abs(self.att_weights2).mean())
            output = (torch.matmul(output, self.att_weights2)) + self.att_bias
            #print(torch.std(output,0))
        else:
            output = torch.matmul(input, self.att_weights) + self.att_bias
        return output

    def update_phi_gradient(self, f1, f2):
        # only handle first part of phi's gradient
        # input f1 should be 128
        #print(f1.shape)
        # print('ushape')
        # print(self.u.shape)
        #flag
        if self.opt.dptype:
            f1 = f1.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            f2 = f2.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            #print('f1test', f1)
            k = self.opt.k
            if self.opt.ar:
                e = k * (f2 * (1 - 2 * self.u))
            else:
                if self.opt.sparse_arm:
                    e = k * ((f1 - f2) * (self.u - .5)) * self.sarm_mask
                else:
                    e = k * ((f1 - f2) * (self.u - .5))
            if self.opt.finetune:
                e[self.test_z[0:1, :, :, :].repeat(self.u.size(0), 1, 1, 1) == 0] = 0
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
                    e[self.test_z[0, :, :, :] == 0] = 0
                self.z_phi.grad = e


    def regularization(self):
        #flag
        if self.opt.finetune:
            return 0.0
        if self.opt.var_dropout:
            return 0
        elif self.opt.dptype:
            #TODO: only for channel-wise attention now.
            # similar with L0 paper

            # if self.opt.hardsigmoid:
            #     pi = F.hardtanh(self.opt.k * self.eta / 7. + .5, 0, 1)
            # else:
            #     pi = torch.sigmoid(self.opt.k * self.eta)
            # l0 = self.lamba * pi.sum() * self.weights.view(-1).size()[0] / self.weights.size(0)
            # wd_col = .5 * self.weight_decay * self.weights.pow(2).sum(3).sum(2).sum(1)
            # wd = torch.sum(pi * wd_col.unsqueeze(0).unsqueeze(2).unsqueeze(3))
            # wb = 0 if not self.use_bias else torch.sum(pi * (.5 * self.weight_decay * self.bias.pow(2).unsqueeze(0).unsqueeze(2).unsqueeze(3)))
            # l2 = wd + wb
            # return l0 + l2
            return 0.0
        else:
            #similar with L0 paper
            # if self.opt.hardsigmoid:
            #     pi = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1)
            # else:
            #     pi = torch.sigmoid(self.opt.k * self.z_phi)

            # l0 = self.lamba * pi.sum() * self.weights.view(-1).size()[0] / self.weights.size(0)
            # wd_col = .5 * self.weight_decay * self.weights.pow(2).sum(3).sum(2).sum(1)
            # wd = torch.sum(pi * wd_col)
            # wb = 0 if not self.use_bias else torch.sum(pi * (.5 * self.weight_decay * self.bias.pow(2)))
            # l2 = wd + wb
            # return l0 + l2
            return 0.0


    def count_expected_flops_and_l0(self):
        '''
        Measures the expected floating point operations (FLOPs) and the expected L0 norm
        '''
        if self.opt.hardsigmoid:
            ppos = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1).sum()
        else:
            ppos = torch.sigmoid(self.opt.k * self.z_phi).sum()

        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels  # vector_length
        flops_per_instance = n + (n - 1)  # (n: multiplications and n-1: additions)

        num_instances_per_filter = ((self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[
            0]) + 1  # for rows
        num_instances_per_filter *= ((self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[
            1]) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_l0 = n * ppos
        if self.use_bias:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

        if not self.opt.ar:
            expected_flops *= 2
            expected_l0 *= 2

        return expected_flops.data, expected_l0.data

    #flag
    def contextual_dropout(self, input):
        if self.opt.dptype:
            if self.opt.dpcha:
                if self.opt.se:
                  
                        # input shape: batch, channel, weight, height
                        # output shape: batch, 1, weight, height
                        input = torch.mean(input, dim=2, keepdim=True)
                        input = torch.mean(input, dim=3, keepdim=True)
                        z_phi = self.ctdo_linear1(torch.transpose(input, 1, 3))
                        #z_phi = self.ctdo_linear1(torch.transpose(input, 1, 3))
                        z_phi = torch.transpose(z_phi, 1, 3)
                        # print('zphi')
                        # #print(torch.sigmoid(z_phi))
                        # print('mean', torch.mean(torch.sigmoid(z_phi)))
                        # print('std', torch.std(torch.sigmoid(z_phi)))
                        return z_phi
                else:
        
                        # input shape: batch, channel, weight, height
                        # output shape: batch, 1, weight, height
                        input = torch.mean(input, dim=2, keepdim=True)
                        input = torch.mean(input, dim=3, keepdim=True)
                        z_phi = self.ctdo_linear1(torch.transpose(input.data, 1, 3))
                        #z_phi = self.ctdo_linear1(torch.transpose(input, 1, 3))
                        z_phi = torch.transpose(z_phi, 1, 3)
                        # print('zphi')
                        # #print(torch.sigmoid(z_phi))
                        # print('mean', torch.mean(torch.sigmoid(z_phi)))
                        # print('std', torch.std(torch.sigmoid(z_phi)))
                        return z_phi
            else:
                
                    # input shape: batch, channel, weight, height
                    # output shape: batch, 1, weight, height
                    z_phi = self.ctdo_linear1(torch.transpose(input.data, 1, 3))
                    z_phi = torch.transpose(z_phi, 1, 3)
                    # print('zphi')
                    # #print(torch.sigmoid(z_phi))
                    # print('mean', torch.mean(torch.sigmoid(z_phi)))
                    # print('std', torch.std(torch.sigmoid(z_phi)))
                    return z_phi

    #flag
    def sample_z(self, batch_size=None, input_=None):
        if self.opt.dptype:
            if self.opt.dpcha:
                if self.opt.ctype == "Gaussian":
                    
                        # input: batch, channel, weight, height
                        # output: batch, 1, weight, height
                        # self.z_phi = self.some_func(input_)
                        # if self.layers  = self.layers [i]
                        self.z_phi = self.contextual_dropout(input_)
                        
                        prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0))

                        #print dropout
                        #print('ARMCon', pi.var(), pi.mean())
                        mu_prior = torch.ones((batch_size, self.out_channels, 1, 1)).to(self.device) 
                        mu_post = torch.ones((batch_size, self.out_channels, 1, 1)).to(self.device) 
                        prior_sigma =  torch.sqrt((1-prior_pi)/prior_pi) 
                        
                        self.u = self.floatTensor(batch_size, self.out_channels, 1, 1).normal_(0, 1)
                        u = self.u
                        
                        pi = torch.sigmoid(self.opt.k * self.z_phi)
                        sigma = torch.sqrt((1-pi)/pi)

                        prior_dist = torch.distributions.normal.Normal(mu_prior, prior_sigma)
                        post_dist = torch.distributions.normal.Normal(mu_post, sigma)
                        if self.training:
                            #Reparameterization trick
                            z = mu_post + sigma * u
                            ##For later use when computing KL
                            self.train_z = z  
                        else:
                            if self.opt.test_sample_mode == 'greedy':
                                z = torch.ones((batch_size, self.out_channels, 1, 1)).to(self.device) 
                            else:
                                z = mu_post + sigma * u

                            self.test_z = z
                        self.prior_nll_true = (-prior_dist.log_prob(z)).mean(1).squeeze()
                        self.post_nll_true = (-post_dist.log_prob(z)).mean(1).squeeze()
                        return z.view(batch_size, self.out_channels, 1, 1)
                        
                else:
         
                        # input: batch, channel, weight, height
                        # output: batch, 1, weight, height
                        # self.z_phi = self.some_func(input_)
                        # if self.layers  = self.layers [i]
                        self.z_phi = self.contextual_dropout(input_)
                        if self.opt.hardsigmoid:
                            pi = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1)
                        else:
                            pi = torch.sigmoid(self.opt.k * self.z_phi)
                        # updating
                        if np.random.uniform() >0.9999:
                            print('mean', torch.mean(pi))
                            print('std', torch.std(pi))
                            print('eta', torch.mean(torch.sigmoid(self.eta * self.opt.k)))
                        #print dropout
                        #print('ARMCon', pi.var(), pi.mean())
                        self.pi = pi
                        if self.forward_mode:
                            z = self.floatTensor(batch_size, self.out_channels, 1, 1).zero_()
                            if self.training:
                                if self.local_rep:
                                    self.u = self.floatTensor(self.out_channels).uniform_(0, 1).expand(batch_size, self.out_channels,1,1)
                                else:
                                    self.u = self.floatTensor(batch_size, self.out_channels, 1, 1).uniform_(0, 1)
                                if self.opt.optim_method:
                                    pi = torch.sigmoid(self.opt.k * self.z_phi)
                                    # self.pi = pi
                                    # if np.random.uniform(0, 1) > 0.99999:
                                    #     print('pi', pi.mean(), pi.std())
                                    # print('pi', pi.mean(), pi.std())
                                    eps = 1e-7
                                    temp = self.concrete_temp
                                    u = self.u
                                    z = (torch.log(pi + eps) - torch.log(1 - pi + eps) + torch.log(u + eps) - torch.log(1 - u + eps))
                                    z = torch.sigmoid(z / temp)
                                    # print('z', z.mean())
                                else:
                                    z[self.u < pi] = 1
                                    if self.opt.use_t_in_training:
                                        z[pi < self.opt.t] = 0
                                self.train_z = z
                                if self.opt.finetune:
                                    z[self.test_z[0:1, :,:,:].repeat(batch_size, 1, 1, 1) == 0] = 0
                                self.post_nll_true = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                                # self.post_nll_true = self.post_nll_true.sum(1).squeeze()
                                self.post_nll_true = self.post_nll_true.mean(1).squeeze()
                                prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0).unsqueeze(2).unsqueeze(2))
                                self.prior_nll_true = - (z * torch.log(prior_pi + epsilon) +(1 - z) * torch.log(1 - prior_pi + epsilon))
                                # self.prior_nll_true = self.prior_nll_true.sum(1).squeeze()
                                self.prior_nll_true = self.prior_nll_true.mean(1).squeeze()

                                self.post_nll_true_rb = - (pi * torch.log(pi + epsilon) + (1 - pi) * torch.log(1 - pi + epsilon))
                                # self.post_nll_true_rb = self.post_nll_true_rb.sum(1).squeeze()
                                self.post_nll_true_rb = self.post_nll_true_rb.mean(1).squeeze()
                                self.prior_nll_true_rb = - (pi * torch.log(prior_pi + epsilon) +(1 - pi) * torch.log(1 - prior_pi + epsilon))
                                # self.prior_nll_true_rb = self.prior_nll_true_rb.sum(1).squeeze()
                                self.prior_nll_true_rb = self.prior_nll_true_rb.mean(1).squeeze()

                            else:
                                if self.opt.test_sample_mode == 'greedy':
                                    z[self.z_phi > 0] = 1
                                    #self.test_z = z
                                    # z = torch.sigmoid(self.z_phi.data).expand(batch_size, self.dim_z)
                                    if self.opt.use_t_in_testing:
                                        z = pi
                                        if self.opt.use_uniform_mask:
                                            if self.opt.finetune:
                                                z[self.test_z[0:1, :,:,:].repeat(batch_size, 1, 1, 1) == 0] = 0
                                            else:
                                                prior_mask = (torch.sigmoid(self.opt.k * self.eta) < self.opt.t).unsqueeze(0).repeat(batch_size, 1).unsqueeze(2).unsqueeze(3)
                                                z[prior_mask] = 0
                                                self.test_z = 1 - prior_mask
                                        else:
                                            z[z < self.opt.t] = 0
                                            self.test_z = z
                                else:
                                    self.u = self.floatTensor(batch_size, self.out_channels, 1, 1).uniform_(0, 1)
                                    z[self.u < pi] = 1
                                    self.test_z = z
                                    if self.opt.use_uniform_mask:
                                        prior_mask = (torch.sigmoid(self.opt.k * self.eta) < self.opt.t).unsqueeze(0).repeat(batch_size, 1).unsqueeze(2).unsqueeze(3)
                                        z[prior_mask] = 0
                                        self.test_z = 1 - prior_mask
                                pi = torch.sigmoid(self.opt.k * self.z_phi)
                                self.post_nll_true = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                                # self.post_nll_true = self.post_nll_true.sum(1).squeeze()
                                self.post_nll_true = self.post_nll_true.mean(1).squeeze()
                                prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0).unsqueeze(2).unsqueeze(2))
                                self.prior_nll_true = - (z * torch.log(prior_pi + epsilon) +(1 - z) * torch.log(1 - prior_pi + epsilon))
                                # self.prior_nll_true = self.prior_nll_true.sum(1).squeeze()
                                self.prior_nll_true = self.prior_nll_true.mean(1).squeeze()

                        else:
                            # pi2 = torch.sigmoid(-self.opt.k * self.z_phi)
                            pi2 = 1 - pi

                            if self.u is None:
                                raise Exception('Forward pass first')
                            z = self.floatTensor(batch_size, self.out_channels, 1, 1).zero_()
                            z[self.u > pi2] = 1
                            if self.opt.finetune:
                                z[self.test_z[0:1,:,:,:].repeat(batch_size, 1,1,1) == 0] = 0
                                # self.test_z = z
                            z_true = self.floatTensor(batch_size, self.out_channels, 1, 1).zero_()
                            z_true[self.u < pi] = 1
                            if self.opt.use_t_in_training:
                                z[pi < self.opt.t] = 0
                            self.sarm_mask = pi.new_ones(batch_size, self.out_channels, 1, 1)
                            self.sarm_mask[z == z_true] = 0.0
                            self.new_pseudo = self.sarm_mask.sum() != 0
                            self.post_nll_sudo = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                            self.post_nll_sudo = self.post_nll_sudo.mean(1).squeeze()
                            prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0).unsqueeze(2).unsqueeze(2))
                            self.prior_nll_sudo = - (z * torch.log(prior_pi + epsilon) +
                                                    (1 - z) * torch.log(1 - prior_pi + epsilon))
                            self.prior_nll_sudo = self.prior_nll_sudo.mean(1).squeeze()
                            self.post_nll_sudo_rb = - (pi * torch.log(pi + epsilon) + (1 - pi) * torch.log(1 - pi + epsilon))
                            self.post_nll_sudo_rb = self.post_nll_sudo_rb.mean(1).squeeze()
                            self.prior_nll_sudo_rb = - (pi * torch.log(prior_pi + epsilon) +
                                                    (1 - pi) * torch.log(1 - prior_pi + epsilon))
                            self.prior_nll_sudo_rb = self.prior_nll_sudo_rb.mean(1).squeeze()
                        return z.view(batch_size, self.out_channels, 1, 1)                

            else:
             
                    # input: batch, channel, weight, height
                    # output: batch, 1, weight, height
                    #self.z_phi = self.some_func(input_)
                    #if self.layers  = self.layers [i]
                    self.z_phi = self.contextual_dropout(input_)
                    if self.opt.hardsigmoid:
                        pi = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1).detach()
                    else:
                        pi = torch.sigmoid(self.opt.k * self.z_phi).detach()
                    #updating
                    #print('mean', torch.mean(pi))
                    #print('std', torch.std(pi))
                    self.pi = pi
                    if self.forward_mode:
                        z = self.floatTensor(batch_size, 1, input_.shape[2], input_.shape[3]).zero_()
                        if self.training:
                            if self.local_rep:
                                self.u = self.floatTensor(self.dim_z).uniform_(0, 1).expand(batch_size, self.dim_z)
                            else:
                                self.u = self.floatTensor(batch_size, 1, input_.shape[2], input_.shape[3]).uniform_(0, 1)
                            z[self.u < pi] = 1
                            if self.opt.use_t_in_training:
                                z[pi < self.opt.t] = 0
                            self.train_z = z
                        else:
                            if self.opt.test_sample_mode == 'greedy':
                                z[self.z_phi > 0] = 1
                                # z = torch.sigmoid(self.z_phi.data).expand(batch_size, self.dim_z)
                                if self.opt.use_t_in_testing:
                                    z = pi
                                    z[z < self.opt.t] = 0
                                self.test_z = z
                            else:
                                self.u = self.floatTensor(batch_size, 1, input_.shape[2], input_.shape[3]).uniform_(0, 1)
                                z[self.u < pi] = 1
                                if self.opt.use_t_in_testing:
                                    z[pi < self.opt.t] = 0
                                self.test_z = z
                        pi = torch.sigmoid(self.opt.k * self.z_phi)
                        self.post_nll_true = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                        # self.post_nll_true = self.post_nll_true.sum(1).squeeze()
                        self.post_nll_true = self.post_nll_true.mean(1).squeeze()
                        prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0))
                        self.prior_nll_true = - (
                                    z * torch.log(prior_pi + epsilon) + (1 - z) * torch.log(1 - prior_pi + epsilon))
                        # self.prior_nll_true = self.prior_nll_true.sum(1).squeeze()
                        self.prior_nll_true = self.prior_nll_true.mean(1).squeeze()

                    else:
                        # pi2 = torch.sigmoid(-self.opt.k * self.z_phi)
                        pi2 = 1 - pi

                        if self.u is None:
                            raise Exception('Forward pass first')
                        z = self.floatTensor(batch_size, 1, input_.shape[2], input_.shape[3]).zero_()
                        z[self.u > pi2] = 1
                        if self.opt.use_t_in_training:
                            z[pi < self.opt.t] = 0
                    return z.view(batch_size, 1, input_.shape[2], input_.shape[3])
        else:
    
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
                #print('mean', torch.mean(pi))
                #print('std', torch.std(pi))
                # print dropout
                #print('ARMCon', pi.var(), pi.mean())
                if self.forward_mode:
                    z = self.floatTensor(batch_size, self.dim_z).zero_()
                    if self.training:
                        if self.local_rep:
                            self.u = self.floatTensor(self.dim_z).uniform_(0, 1).expand(batch_size, self.dim_z)
                        else:
                            self.u = self.floatTensor(batch_size, self.dim_z).uniform_(0, 1)

                        if self.opt.gumbelconcrete:
                            pi = torch.sigmoid(self.opt.k * self.z_phi)
                            eps = 1e-7
                            temp = self.concrete_temp
                            u = self.u
                            z = (torch.log(pi + eps) - torch.log(1 - pi + eps) + torch.log(u + eps) - torch.log(
                                1 - u + eps))
                            z = torch.sigmoid(z / temp)
                        else:
                            z[self.u < pi.expand(batch_size, self.dim_z)] = 1
                        if self.opt.use_t_in_training:
                            z[(pi.expand(batch_size, self.dim_z)) < self.opt.t] = 0
                        self.train_z = z
                        if self.opt.finetune:
                            z[self.test_z[0:1,:,:,:].repeat(batch_size, 1,1,1) == 0] = 0
                        if self.opt.concretedp:

                            pi = torch.sigmoid(self.opt.k * self.z_phi)
                            self.post_nll_true = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                            self.post_nll_true = self.post_nll_true.mean(1).squeeze()
                            # print('test', self.post_nll_true.shape)


                            prior_pi = torch.sigmoid(self.opt.k * self.eta)
                            self.prior_nll_true = - (z * torch.log(prior_pi + epsilon) + (1 - z) * torch.log(1 - prior_pi + epsilon))
                            self.prior_nll_true = self.prior_nll_true.mean(1).squeeze()
                            # print('test', self.prior_nll_true.shape)
                    else:
                        if self.opt.test_sample_mode == 'greedy':
                            z[self.z_phi.expand(batch_size, self.dim_z) > 0] = 1
                            # z = torch.sigmoid(self.z_phi.data).expand(batch_size, self.dim_z)
                            if self.opt.use_t_in_testing:
                                z = pi.expand(batch_size, self.dim_z)
                                if self.opt.finetune:
                                    z[self.test_z[0:1, :, :, :].repeat(batch_size, 1, 1, 1) == 0] = 0
                                else:
                                    z[z < self.opt.t] = 0
                                    self.test_z = z
                        else:
                            self.u = self.floatTensor(batch_size, self.dim_z).uniform_(0, 1)
                            z[self.u < pi.expand(batch_size, self.dim_z)] = 1
                            if self.opt.finetune:
                                z[self.test_z[0:1, :, :, :].repeat(batch_size, 1, 1, 1) == 0] = 0
                            else:
                                z[pi.expand(batch_size, self.dim_z) < self.opt.t] = 0
                            #self.test_z =  (pi.expand(batch_size, self.dim_z) < self.opt.t).type_as(z)
                        if self.opt.concretedp:
                            pi = torch.sigmoid(self.opt.k * self.z_phi)
                            self.post_nll_true = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                            # self.post_nll_true = self.post_nll_true.sum(1).squeeze()
                            self.post_nll_true = self.post_nll_true.mean(1).squeeze()
                            prior_pi = torch.sigmoid(self.opt.k * self.eta.unsqueeze(0))
                            self.prior_nll_true = - (
                                        z * torch.log(prior_pi + epsilon) + (1 - z) * torch.log(1 - prior_pi + epsilon))
                            # self.prior_nll_true = self.prior_nll_true.sum(1).squeeze()
                            self.prior_nll_true = self.prior_nll_true.mean(1).squeeze()
                else:
                    # pi2 = torch.sigmoid(-self.opt.k * self.z_phi)
                    pi2 = 1 - pi
                    if self.u is None:
                        raise Exception('Forward pass first')
                    z = self.floatTensor(batch_size, self.dim_z).zero_()
                    z[self.u > pi2.expand(batch_size, self.dim_z)] = 1
                    if self.opt.finetune:
                        z[self.test_z[0:1, :, :, :].repeat(batch_size, 1, 1, 1) == 0] = 0
                    if self.opt.use_t_in_training:
                        z[pi.expand(batch_size, self.dim_z) < self.opt.t] = 0
                    if self.opt.concretedp:
                        pi = torch.sigmoid(self.opt.k * self.z_phi)
                        self.post_nll_sudo = - (z * torch.log(pi + epsilon) + (1 - z) * torch.log(1 - pi + epsilon))
                        self.post_nll_sudo = self.post_nll_sudo.mean(1).squeeze()
                        prior_pi = torch.sigmoid(self.opt.k * self.eta)
                        self.prior_nll_sudo = - (z * torch.log(prior_pi + epsilon) +(1 - z) * torch.log(1 - prior_pi + epsilon))
                        self.prior_nll_sudo = self.prior_nll_sudo.mean(1).squeeze()
                return z.view(batch_size, self.dim_z, 1, 1)


    def sample_gauss(self, batch_size):
        if self.opt.hardsigmoid:
            pi = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1).detach()
        else:
            pi = torch.sigmoid(self.opt.k * self.z_phi).detach()
        self.pi = pi
        if self.local_rep:
            self.u = self.floatTensor(self.dim_z).normal_(1, self.opt.sd).expand(batch_size, self.dim_z)
        else:
            self.u = self.floatTensor(batch_size, self.dim_z).normal_(1, self.opt.sd)
        z = self.u
        self.train_z = z
        self.test_z = z
        return z.view(batch_size, self.dim_z, 1, 1)
    
    def conv2d_train(self, input_, eps = 1e-8):
        
        if self.clip_alpha is not None:
            log_alpha = torch.clamp(2.0 * self.log_sigma - torch.log(self.weights**2 + eps), max = self.clip_alpha)
            log_sigma2 = log_alpha + torch.log(self.weights**2 + eps)

        b = None if not self.use_bias else self.bias
        mu_activation = F.conv2d(input_, self.weights, b, self.stride, self.padding, self.dilation)

        std_activation = torch.sqrt(F.conv2d(input_**2, torch.exp(log_sigma2), None, self.stride, self.padding, self.dilation) + eps)
        return mu_activation + std_activation * self.floatTensor(std_activation.size()).normal_(0, 1)

    def conv2d_eval(self, input_,  threshold = 3.0, eps=1e-8):
        log_alpha = 2.0 * self.log_sigma - torch.log(self.weights**2 + eps)
        masked_weights = (log_alpha <= threshold).float() * self.weights
        self.sparsity_ratio = (log_alpha > threshold).float().mean()
        b = None if not self.use_bias else self.bias

        if self.opt.test_sample_mode == 'greedy':
            return F.conv2d(input_, masked_weights, b, self.stride, self.padding, self.dilation)
        else:
            if self.clip_alpha is not None:
                log_alpha = torch.clamp(2.0 * self.log_sigma - torch.log(self.weights**2 + eps), -self.clip_alpha, self.clip_alpha)
                log_sigma2 = log_alpha + torch.log(self.weights**2 + eps)

            mu_activation = F.conv2d(input_, self.weights, b, self.stride, self.padding, self.dilation)

            std_activation = torch.sqrt(F.conv2d(input_**2, torch.exp(log_sigma2), None, self.stride, self.padding, self.dilation) + eps)
            return mu_activation + std_activation * self.floatTensor(std_activation.size()).normal_(0, 1)
    def get_kl(self, eps=1e-8):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        c = -k1
        log_alpha = 2.0 * self.log_sigma - torch.log(self.weights**2 + eps)
        term_1 = k1 * torch.sigmoid(k2 + k3 * log_alpha)
        term_2 = -0.5 * torch.log(1 + torch.exp(-(log_alpha)))
        neg_dkl = term_1 + term_2 + c
        return -neg_dkl.sum()
    def variational_dropout(self, input_):
        if self.training:
            return self.conv2d_train(input_)
        return self.conv2d_eval(input_)

    def forward(self, input_):
        """ forward for fc """
        #updating
        # flag
        if self.opt.GFFN_dropout:
            if self.input_shape is None:
                self.input_shape = input_.size()
            b = None if not self.use_bias else self.bias
            output = F.conv2d(input_, self.weights, b, self.stride, self.padding, self.dilation)
            return output

        elif self.opt.var_dropout:
            output = self.variational_dropout(input_)
            return output

        elif self.opt.dptype:
            if self.input_shape is None:
                self.input_shape = input_.size()
            b = None if not self.use_bias else self.bias
            output = F.conv2d(input_, self.weights, b, self.stride, self.padding, self.dilation)
            if self.opt.se:
                self.z_phi = self.contextual_dropout(output)
                if self.opt.hardsigmoid:
                    pi = F.hardtanh(self.opt.k * self.z_phi / 7. + .5, 0, 1)
                else:
                    pi = torch.sigmoid(self.opt.k * self.z_phi)
                pi.view(pi.size(0), self.out_channels, 1, 1)
                z = pi
                self.train_z = z
                self.test_z = z
                self.pi =pi.data
            else:
                z = self.sample_z(batch_size=output.size(0), input_=output)
               
            output = output.mul(z) # batch, channel, weight, height, * batch, 1, weight, height
            return output
        else:
            """ forward for fc """
            if self.opt.dropout_distribution == 'bernoulli':
                if self.input_shape is None:
                    self.input_shape = input_.size()
                b = None if not self.use_bias else self.bias
                output = F.conv2d(input_, self.weights, b, self.stride, self.padding, self.dilation)
                z = self.sample_z(batch_size=output.size(0))
                output = output.mul(z)
            elif self.opt.dropout_distribution == 'gaussian':
                if self.input_shape is None:
                    self.input_shape = input_.size()
                b = None if not self.use_bias else self.bias
                output = F.conv2d(input_, self.weights, b, self.stride, self.padding, self.dilation)
                if self.training or self.opt.test_sample_mode != 'greedy':
                    z = self.sample_gauss(output.size(0))
                    output = output.mul(z)
            return output


    def activated_neurons(self):
        return (self.test_z > 0).sum() / self.test_z.size(0)

    def expected_activated_neurons(self):
        return (self.train_z > 0).sum() / self.train_z.size(0)

    def masked_weight(self):
        return self.weights * self.test_z[0].reshape(self.out_channels, 1, 1, 1)

    def __repr__(self):
        s = (
            '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, lamba={lamba}, weight_decay={weight_decay}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

