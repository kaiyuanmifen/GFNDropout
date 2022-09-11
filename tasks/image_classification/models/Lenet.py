from copy import deepcopy
import torch
import torch.nn as nn
from models.layer import ArmConv2d
from models.layer import ARMDense
import numpy as np
from config import opt
#updating
import math
import torch.nn.functional as F


def get_flat_fts(in_size, fts):
    dummy_input = torch.ones(1, *in_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    f = fts(torch.autograd.Variable(dummy_input))
    return int(np.prod(f.size()[1:]))


class ARMLeNet5(nn.Module):
    def __init__(self, num_classes=10, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500,
                 N=60000, beta_ema=0.999, weight_decay=0.0005, lambas=(.1, .1, .1, .1)):
        super(ARMLeNet5, self).__init__()
        self.N = N
        assert (len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay
        self.epoch = 0

        convs = [ArmConv2d(input_size[0], conv_dims[0], 5, droprate_init=opt.lenet_dr, lamba=lambas[0],
                           local_rep=opt.local_rep,
                           weight_decay=self.weight_decay),
                 nn.ReLU(), nn.MaxPool2d(2),
                 ArmConv2d(conv_dims[0], conv_dims[1], 5, droprate_init=opt.lenet_dr, lamba=lambas[1],
                           local_rep=opt.local_rep,
                           weight_decay=self.weight_decay),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs_list = convs
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [ARMDense(flat_fts, self.fc_dims, droprate_init=opt.lenet_dr, lamba=lambas[2], local_rep=opt.local_rep,
                        weight_decay=self.weight_decay), nn.ReLU(),
               ARMDense(self.fc_dims, num_classes, droprate_init=opt.lenet_dr, lamba=lambas[3], local_rep=opt.local_rep,
                        weight_decay=self.weight_decay)]
        self.fcs_list = fcs
        self.fcs = nn.Sequential(*fcs)
        self.layers = []
        for m in self.modules():
            if isinstance(m, ARMDense) or isinstance(m, ArmConv2d):
                self.layers.append(m)
                #print.self.layers
        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def update_phi_gradient(self, f1, f2):
        # previous: f1 128, f2, 128
        # now: f1 a list of 4 128, f2, 128, 4
        #flag
        if opt.dptype:
            if not opt.se:
                for i, layer in enumerate(self.layers):
                    # f1 = torch.from_numpy(f1)
                    # print(f1[1])
                    layer.update_phi_gradient(f1[i], f2)
        else:
            for layer in self.layers:
                layer.update_phi_gradient(f1, f2)



    def forward_mode(self, mode):
        # mode is a list now
        #mode = []
        #flag
        if opt.dptype:
            for i, layer in enumerate(self.layers):
                layer.forward_mode = mode[i]
        else:
            for layer in self.layers:
                layer.forward_mode = mode



    def score(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        o = self.fcs(o)
        return o

    def forward(self, x, y=None):
        #flag
        if opt.dptype:
            if self.training:
                if opt.optim_method or opt.se:
                    if opt.lambda_kl != 0.0 and not opt.se:
                        f_kl=0
                        f_prior = 0
                        if torch.cuda.is_available():
                            self.convs_list[0] = self.convs_list[0].cuda()
                            self.convs_list[3] = self.convs_list[3].cuda()
                            self.fcs_list[0] = self.fcs_list[0].cuda()
                            self.fcs_list[2] = self.fcs_list[2].cuda()
                        self.forward_mode([True] * len(self.layers))
                        score = self.score(x)
                        f_kl = f_kl + self.convs_list[0].post_nll_true - self.convs_list[0].prior_nll_true
                        f_prior = f_prior + self.convs_list[0].prior_nll_true
                        f_kl = f_kl + self.convs_list[3].post_nll_true - self.convs_list[3].prior_nll_true
                        f_prior = f_prior + self.convs_list[3].prior_nll_true
                        f_kl = f_kl + self.fcs_list[0].post_nll_true - self.fcs_list[0].prior_nll_true
                        f_prior = f_prior + self.fcs_list[0].prior_nll_true
                        f_kl = f_kl + self.fcs_list[2].post_nll_true - self.fcs_list[2].prior_nll_true
                        f_prior = f_prior + self.fcs_list[2].prior_nll_true
                        #f_kl = f_kl.unsqueeze(1)
                        # if opt.learn_prior:
                        #     f_prior.mean().backward(retain_graph = True)


                        kl_loss = (- opt.lambda_kl * f_kl).mean()
                        kl_loss.backward(retain_graph = True)

                    else:
                        self.forward_mode([True] * len(self.layers))
                        score = self.score(x)
                else:
                    f1app = []
                    f2_kl = 0
                    f2_prior = 0
                    update_flag = []
                    # first layer
                    if torch.cuda.is_available():
                        self.convs_list[0] = self.convs_list[0].cuda()
                        self.convs_list[3] = self.convs_list[3].cuda()
                        self.fcs_list[0] = self.fcs_list[0].cuda()
                        self.fcs_list[2] = self.fcs_list[2].cuda()
                    # true actions
                    self.forward_mode([True] * len(self.layers))
                    main_traj = self.convs_list[0](x)
                    main_traj = nn.ReLU()(main_traj)
                    main_traj = nn.MaxPool2d(2)(main_traj)
                    f1_kl = f2_kl
                    f2_kl = f2_kl + self.convs_list[0].post_nll_true - self.convs_list[0].prior_nll_true
                    f2_prior = f2_prior + self.convs_list[0].prior_nll_true
                    # pseudo actions
                    self.forward_mode([False] * len(self.layers))
                    pseudo_traj = self.convs_list[0](x).clone()
                    pseudo_traj = nn.ReLU()(pseudo_traj)
                    pseudo_traj = nn.MaxPool2d(2)(pseudo_traj)
                    f1_kl = f1_kl + self.convs_list[0].post_nll_sudo - self.convs_list[0].prior_nll_sudo
                    self.forward_mode([True] * len(self.layers))
                    pseudo_traj = self.convs_list[3](pseudo_traj)
                    pseudo_traj = nn.ReLU()(pseudo_traj)
                    pseudo_traj = nn.MaxPool2d(2)(pseudo_traj)
                    f1_kl = f1_kl + self.convs_list[3].post_nll_true - self.convs_list[3].prior_nll_true
                    pseudo_traj = pseudo_traj.view(pseudo_traj.size(0), -1)
                    pseudo_traj = self.fcs(pseudo_traj)
                    #print('lol', f1_kl.shape, self.fcs_list[0].post_nll_true.shape)
                    f1_kl = f1_kl + self.fcs_list[0].post_nll_true - self.fcs_list[0].prior_nll_true
                    f1_kl = f1_kl + self.fcs_list[2].post_nll_true - self.fcs_list[2].prior_nll_true

                    f1 = nn.CrossEntropyLoss(reduce=False)(pseudo_traj, y).data - opt.lambda_kl * f1_kl.data
                    f1 = f1 / f1.size(0)
                    f1app.append(f1)
                    # second layer
                    f1_kl = f2_kl
                    x = main_traj
                    main_traj = self.convs_list[3](x)
                    main_traj = nn.ReLU()(main_traj)
                    main_traj = nn.MaxPool2d(2)(main_traj)
                    f2_kl = f2_kl + self.convs_list[3].post_nll_true - self.convs_list[3].prior_nll_true
                    f2_prior = f2_prior + self.convs_list[3].prior_nll_true
                    # pseudo actions
                    self.forward_mode([False] * len(self.layers))
                    pseudo_traj = self.convs_list[3](x)
                    pseudo_traj = nn.ReLU()(pseudo_traj)
                    pseudo_traj = nn.MaxPool2d(2)(pseudo_traj)
                    f1_kl = f1_kl + self.convs_list[3].post_nll_sudo - self.convs_list[3].prior_nll_sudo

                    self.forward_mode([True] * len(self.layers))
                    pseudo_traj = pseudo_traj.view(pseudo_traj.size(0), -1)
                    pseudo_traj = self.fcs(pseudo_traj)
                    f1_kl = f1_kl + self.fcs_list[0].post_nll_true - self.fcs_list[0].prior_nll_true
                    f1_kl = f1_kl + self.fcs_list[2].post_nll_true - self.fcs_list[2].prior_nll_true
                    f1 = nn.CrossEntropyLoss(reduce=False)(pseudo_traj, y).data - opt.lambda_kl * f1_kl.data
                    f1 = f1 / f1.size(0)
                    f1app.append(f1)
                    # third layer
                    f1_kl = f2_kl
                    x = main_traj
                    x = x.view(x.size(0), -1)
                    main_traj = self.fcs_list[0](x)
                    main_traj = nn.ReLU()(main_traj)
                    f2_kl = f2_kl + self.fcs_list[0].post_nll_true - self.fcs_list[0].prior_nll_true
                    f2_prior = f2_prior + self.fcs_list[0].prior_nll_true
                    # pseudo actions
                    self.forward_mode([False] * len(self.layers))
                    pseudo_traj = self.fcs_list[0](x)
                    pseudo_traj = nn.ReLU()(pseudo_traj)
                    f1_kl = f1_kl + self.fcs_list[0].post_nll_sudo - self.fcs_list[0].prior_nll_sudo
                    self.forward_mode([True] * len(self.layers))
                    pseudo_traj = self.fcs_list[2](pseudo_traj)
                    f1_kl = f1_kl + self.fcs_list[2].post_nll_true - self.fcs_list[2].prior_nll_true
                    f1 = nn.CrossEntropyLoss(reduce=False)(pseudo_traj, y).data - opt.lambda_kl * f1_kl.data
                    f1 = f1 / f1.size(0)
                    f1app.append(f1)
                    # fourth layer
                    f1_kl = f2_kl
                    x = main_traj
                    main_traj = self.fcs_list[2](x)
                    f2_kl = f2_kl + self.fcs_list[2].post_nll_true - self.fcs_list[2].prior_nll_true
                    f2_prior = f2_prior + self.fcs_list[2].prior_nll_true
                    self.forward_mode([False] * len(self.layers))
                    pseudo_traj = self.fcs_list[2](x)
                    f1_kl = f1_kl + self.fcs_list[2].post_nll_sudo - self.fcs_list[2].prior_nll_sudo
                    f1 = nn.CrossEntropyLoss(reduce=False)(pseudo_traj, y).data -opt.lambda_kl * f1_kl.data
                    f1 = f1 / f1.size(0)
                    f1app.append(f1)
                    f2 = nn.CrossEntropyLoss(reduce=False)(main_traj, y).data - opt.lambda_kl * f2_kl.data
                    f2 = f2 / f2.size(0)
                    self.update_phi_gradient(f1app, f2)
                    #print('kl', f1.mean(), f1_kl.mean(), f2_kl.mean(), self.fcs_list[2].post_nll_sudo.mean(), self.fcs_list[2].prior_nll_sudo.mean(),
                    #      self.fcs_list[2].post_nll_true.mean(), self.fcs_list[2].prior_nll_true.mean())
                    if opt.learn_prior:
                        f2_prior = f2_prior
                        f2_prior.mean().backward()
                    score = main_traj
            else:
                self.forward_mode([True] * len(self.layers))
                score = self.score(x)
            return score
        else:
            if self.training:
                self.forward_mode(True)
                score = self.score(x)

                self.eval() if opt.gpus <= 1 else self.module.eval()
                if opt.ar is not True:
                    self.forward_mode(False)
                    score2 = self.score(x).data
                    f1 = nn.CrossEntropyLoss(reduce=False)(score2, y).data
                else:
                    f1 = 0
                f2 = nn.CrossEntropyLoss(reduce=False)(score, y).data

                self.update_phi_gradient(f1, f2)
                self.train() if opt.gpus <= 1 else self.module.train()
            else:
                self.forward_mode(True)
                score = self.score(x)
            return score

    #updating
    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += (1. / self.N) * layer.regularization()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0


    #Updating:
    def get_dprate(self):
        # flag
        dprate = []
        i=0
        for layer in self.layers:
            if i >= 2:
                break
            i += 1
            #dprate.append(torch.mean(layer.pi).cpu().item())
            dprate.append((layer.pi.detach()).cpu().numpy())
        return dprate



    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def get_activated_neurons(self):
        return [layer.activated_neurons() for layer in self.layers]

    def get_expected_activated_neurons(self):
        return [layer.expected_activated_neurons() for layer in self.layers]

    def z_phis(self):
        return [layer.z_phi for layer in self.layers]

    def prune_rate(self):
        '''
        the number of parameters being pruned / the number of parameters
        '''
        l = [layer.activated_neurons().cpu().numpy() for layer in self.layers]
        return 100 - 100.0 * (l[0] * 25.0 + l[1] * l[0] * 25.0 + l[2] * l[3] + l[3] * 10.0) / (
                    20.0 * 25 + 50 * 20 * 25 + 800 * 500 + 5000)
