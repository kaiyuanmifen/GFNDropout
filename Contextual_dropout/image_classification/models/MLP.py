import torch
import torch.nn as nn

from models.layer import ARMDense
import torch.nn.functional as F
from copy import deepcopy
import numpy as np


epsilon = 1e-7
class ARMMLP(nn.Module):
    def __init__(self, input_dim=784, num_classes=10, N=60000, layer_dims=(300, 100), beta_ema=0.999,
                 weight_decay=5e-4, lambas=(.1, .1, .1), local_rep=True,opt=None):
        super(ARMMLP, self).__init__()

        self.opt=opt
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.N = N
        self.weight_decay = N * weight_decay
        self.lambas = lambas
        self.epoch = 0
        self.elbo = 0
        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
            droprate_init, lamba = 0.2 if i == 0 else self.opt.mlp_dr, lambas[i] if len(lambas) > 1 else lambas[0]
            layers += [ARMDense(inp_dim, dimh, droprate_init=droprate_init, weight_decay=self.weight_decay,
                                lamba=lamba, local_rep=self.opt.local_rep,opt=self.opt), nn.ReLU()]

        layers.append(ARMDense(self.layer_dims[-1], num_classes, droprate_init=self.opt.mlp_dr,
                               weight_decay=self.weight_decay, lamba=lambas[-1], local_rep=self.opt.local_rep,opt=self.opt))
        self.output = nn.Sequential(*layers)
        self.layers = []
        for m in self.modules():
            if isinstance(m, ARMDense):
                self.layers.append(m)

        self.beta_ema = beta_ema
        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def score(self, x):
        return self.output(x.view(-1, self.input_dim))

    def cluster_penalty(self, pi, label):
        # pi batch size x features, label, batch size
        global_mean = torch.mean(pi, dim=0, keepdim=True)
        global_center_distance = torch.mean(torch.sqrt(torch.sum(torch.pow(pi - global_mean, 2), dim=1)))
        one_hot_code = F.one_hot(label).type_as(pi)
        degree_inverse = 1.0 / (torch.sum(one_hot_code, dim=0, keepdim=True) + epsilon)
        local_mean = torch.matmul(torch.matmul(one_hot_code * degree_inverse, torch.transpose(one_hot_code, 0, 1)), pi)
        local_center_distance = torch.mean(torch.sqrt(torch.sum(torch.pow(pi - local_mean, 2), 1)))
        penalty = local_center_distance / (global_center_distance+epsilon)
        return penalty

    def forward(self, x, y=None):
        
        if self.opt.var_dropout:
            if self.training:
                score = self.score(x)
                if self.epoch <= self.opt.N_t:
                    beta = (1.0/self.opt.N_t) * self.epoch
                else:
                    beta = 1
                kl = 0
                for i, layer in enumerate(self.layers):
                    kl += layer.get_kl()
                kl_loss = (1.0/self.N) * (self.opt.lambda_kl * beta * kl)
                kl_loss.backward(retain_graph = True)
            else:
                score = self.score(x)
                self.elbo = self.elbo_fun(nn.CrossEntropyLoss(reduce=False)(score, y).data)
            return score
        elif self.opt.dptype:
            # include se and gumbel, and ARM
            if self.training:
                if self.opt.optim_method or self.opt.se:
                    self.forward_mode([True] * len(self.layers))
                    score = self.score(x)
                    if self.opt.lambda_kl != 0.0 and not self.opt.se:
                        f_kl = 0
                        f_prior = 0
                        cluster_loss = 0
                        for i, layer in enumerate(self.layers):
                            if self.opt.rb:
                                f_kl = f_kl + (layer.post_nll_true_rb - layer.prior_nll_true_rb)# * (1-np.exp(-self.epoch * self.opt.kl_anneal_rate))
                                f_prior = f_prior + layer.prior_nll_true_rb
                            else:
                                f_kl = f_kl + (layer.post_nll_true - layer.prior_nll_true) #* (1- np.exp(-self.epoch * self.opt.kl_anneal_rate))
                                f_prior = f_prior + layer.prior_nll_true
                            if self.opt.cluster_penalty != 0.0:
                                cluster_loss = cluster_loss + np.exp(-self.epoch * self.opt.cp_anneal_rate) * self.opt.cluster_penalty * self.cluster_penalty(layer.pi, y)
                        if self.opt.cluster_penalty != 0.0:
                            cluster_loss.backward(retain_graph=True)
                            if np.random.uniform() > 0.99:
                                print('cluster_loss', cluster_loss / (np.exp(-self.epoch * self.opt.cp_anneal_rate) * self.opt.cluster_penalty * len(self.layers)))
                            #print('cluster_loss', cluster_loss)
                        # if self.opt.learn_prior:
                        #     f_prior.mean().backward(retain_graph = True)
                        kl_loss = (- self.opt.lambda_kl * f_kl).mean()
                        #print('kl_loss', kl_loss)
                        kl_loss.backward(retain_graph = True)
                else:
                    if self.opt.ctype == "Gaussian":
                        self.forward_mode([True] * len(self.layers))
                        score = self.score(x)
                        f_kl = 0
                        #f_prior = 0
                        for i, layer in enumerate(self.layers):
                            f_kl = f_kl + (layer.post_nll_true - layer.prior_nll_true)
                            #f_prior = f_prior + layer.prior_nll_true
                        kl_loss = (- self.opt.lambda_kl * f_kl).mean()
                        #if self.opt.learn_prior:
                        #    f_prior.mean().backward(retain_graph = True)
                        kl_loss.backward(retain_graph = True)
                    else:
                        f1app = []
                        f2_kl = 0
                        f2_kl_rb = 0
                        f2_prior = 0
                        update_flag = []
                        out = x.view(-1, self.input_dim)
                        for i in range(len(self.layers)):
                            # true actions
                            self.forward_mode([True] * len(self.layers))
                            main_traj = self.layers[i](out)
                            f1_kl = f2_kl
                            f2_kl = f2_kl + self.layers[i].post_nll_true - self.layers[i].prior_nll_true
                            f2_prior = f2_prior + self.layers[i].prior_nll_true
                            f2_kl_rb = f2_kl_rb + self.layers[i].post_nll_true_rb - self.layers[i].prior_nll_true_rb
                            # pseudo actions
                            self.forward_mode([False] * len(self.layers))
                            pseudo_traj = self.layers[i](out).clone()
                            if i < len(self.layers) - 1:
                                main_traj = nn.ReLU()(main_traj)
                                pseudo_traj = nn.ReLU()(pseudo_traj)
                            f1_kl = f1_kl + self.layers[i].post_nll_sudo - self.layers[i].prior_nll_sudo
                            if self.layers[i].new_pseudo:
                                self.forward_mode([True] * len(self.layers))
                                for k in range(i+1, len(self.layers)):
                                    pseudo_traj = self.layers[k](pseudo_traj)
                                    if k < len(self.layers) - 1:
                                        pseudo_traj = nn.ReLU()(pseudo_traj)
                                    f1_kl = f1_kl + self.layers[k].post_nll_true - self.layers[k].prior_nll_true
                                pseudo_score = pseudo_traj
                                f1 = nn.CrossEntropyLoss(reduce=False)(pseudo_score, y).data - self.opt.lambda_kl * f1_kl.data
                                f1 = f1 / f1.size(0)
                                f1app.append(f1)
                                update_flag.append(True)
                            else:
                                f1app.append(0.0)
                                update_flag.append(False)
                            out = main_traj
                        score = out
                        f2 = nn.CrossEntropyLoss(reduce=False)(score, y).data - self.opt.lambda_kl * f2_kl.data
                        f2 = f2 / f2.size(0)
                        self.update_phi_gradient(f1app, f2, update_flag)
                        if self.opt.learn_prior:
                            f2_prior.mean().backward(retain_graph = True)
                        # if self.opt.rb:
                        #     f2_kl_rb = f2_kl_rb
                        #     #print(f2_kl_rb)
                        #     f2_kl_rb.mean().backward(retain_graph = True)
                        #TODO: write rb for ARM.
            else:
                self.forward_mode([True] * len(self.layers))
                score = self.score(x)
                self.elbo = self.elbo_fun(nn.CrossEntropyLoss(reduce=False)(score, y).data)
            return score
        else:
            if self.opt.concretedp:
                if self.opt.gumbelconcrete:
                    if self.training:
                        self.forward_mode(True)
                        score = self.score(x)
                        if self.opt.lambda_kl != 0.0:
                            f_kl = 0
                            f_prior = 0
                            for i, layer in enumerate(self.layers):
                                if self.opt.rb:
                                    f_kl = f_kl + (layer.post_nll_true_rb - layer.prior_nll_true_rb)# * (1-np.exp(-self.epoch * self.opt.kl_anneal_rate))
                                    f_prior = f_prior + layer.prior_nll_true_rb
                                else:
                                    f_kl = f_kl + (layer.post_nll_true - layer.prior_nll_true) #* (1- np.exp(-self.epoch * self.opt.kl_anneal_rate))
                                    f_prior = f_prior + layer.prior_nll_true
                                #print('cluster_loss', cluster_loss)
                            # if self.opt.learn_prior:
                            #     f_prior.mean().backward(retain_graph = True)
                            #print('test', f_kl)
                            #TODO: replace 60000 with the size of training data. Note z is a global variable.
                            kl_loss = (- self.opt.lambda_kl * (1.0 / 60000.0) * f_kl).mean()
                            #print('kl_loss', kl_loss)
                            kl_loss.backward(retain_graph = True)
                    else:
                        self.forward_mode(True)
                        score = self.score(x)
                        self.elbo = self.elbo_fun(nn.CrossEntropyLoss(reduce=False)(score, y).data)
                else:
                    if self.training:
                        self.forward_mode(True)
                        score = self.score(x)
                        f1_kl = 0
                        f2_kl = 0
                        f1_prior =0
                        f2_prior = 0
                        for i in range(len(self.layers)):
                            self.forward_mode(True)
                            f2_kl = f2_kl + self.layers[i].post_nll_true - self.layers[i].prior_nll_true
                            f2_prior = f2_prior + self.layers[i].prior_nll_true

                        self.eval() if self.opt.gpus <= 1 else self.module.eval()
                        if self.opt.ar is not True:
                            self.forward_mode(False)
                            score2 = self.score(x).data
                            f1 = nn.CrossEntropyLoss(reduce=False)(score2, y).data
                            f1 = f1/f1.size(0)
                            for i in range(len(self.layers)):
                                f1_kl = f1_kl + self.layers[i].post_nll_sudo - self.layers[i].prior_nll_sudo
                                f1_prior = f1_prior + self.layers[i].prior_nll_sudo
                        else:
                            f1 = 0
                        f2 = nn.CrossEntropyLoss(reduce=False)(score, y).data - self.opt.lambda_kl * (1.0 / 60000.0)* f2_kl.data
                        f1 = f1 - self.opt.lambda_kl * (1.0 / 60000.0)* f1_kl.data
                        f2 = f2/f2.size(0)
                        f1 = f1/f1.size(0)
                        self.update_phi_gradient(f1, f2)
                        self.train() if self.opt.gpus <= 1 else self.module.train()
                    else:
                        self.forward_mode(True)
                        score = self.score(x)
                        self.elbo = self.elbo_fun(nn.CrossEntropyLoss(reduce=False)(score, y).data)

            else:
                if self.training:
                    self.forward_mode(True)
                    score = self.score(x)
                    if not self.opt.fixdistrdp:
                        self.eval() if self.opt.gpus <= 1 else self.module.eval()
                        if self.opt.ar is not True:
                            self.forward_mode(False)
                            score2 = self.score(x).data
                            f1 = nn.CrossEntropyLoss(reduce=False)(score2, y).data
                        else:
                            f1 = 0
                        f2 = nn.CrossEntropyLoss(reduce=False)(score, y).data

                        self.update_phi_gradient(f1, f2)
                        self.train() if self.opt.gpus <= 1 else self.module.train()
                else:
                    self.forward_mode(True)
                    score = self.score(x)
                    self.elbo = -nn.CrossEntropyLoss(reduce=False)(score, y).data.mean()
            return score

    def update_phi_gradient(self, f1, f2, update_flag=None):
        if self.opt.dptype:
            if not self.opt.se:
                if update_flag is not None:
                    for i, layer in enumerate(self.layers):
                        if update_flag[i]:
                            layer.update_phi_gradient(f1[i], f2)
                else:
                    for i, layer in enumerate(self.layers):
                        layer.update_phi_gradient(f1[i], f2)
        else:
            for layer in self.layers:
                layer.update_phi_gradient(f1, f2)

    def elbo_fun(self, nn_loss):
        elbo = nn_loss
        if self.opt.var_dropout:
            elbo = elbo.mean()
            for i in range(len(self.layers)):
                elbo = elbo + (1/self.N) * self.layers[i].get_kl().detach()
            return -elbo
        for i in range(len(self.layers)):
            if self.opt.dptype:
                elbo = elbo -  (self.layers[i].post_nll_true.data - self.layers[i].prior_nll_true.data)
            else:
                elbo = elbo - (1.0 / 60000.0) * (self.layers[i].post_nll_true.data - self.layers[i].prior_nll_true.data)
        return -elbo.mean()

    def forward_mode(self, mode):
        if self.opt.dptype:
            for i, layer in enumerate(self.layers):
                layer.forward_mode = mode[i]
        else:
            for layer in self.layers:
                layer.forward_mode = mode

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

    def get_activated_neurons(self):
        return [layer.activated_neurons() for layer in self.layers]

    def get_expected_activated_neurons(self):
        return [layer.expected_activated_neurons() for layer in self.layers]

    def prune_rate(self):
        if self.opt.var_dropout:
            return 0
        l = [layer.activated_neurons().cpu().numpy() for layer in self.layers]
        if self.opt.dptype:
            pruning_rate = 100 - 100.0 * ((784.**2  + 300.0**2 + 100.0**2)/ self.opt.cha_factor +
                    l[0] * 300.0 + l[1] * 100.0 + l[2] * 10.) / (784. * 300. + 30000. + 1000.)
            pruning_rate_2 = 100 - 100.0 * (l[0] * 300.0 + l[1] * 100.0 + l[2] * 10.) / (784. * 300. + 30000. + 1000.)
            print('decoder pruning rate', pruning_rate_2)
        else:
            pruning_rate = 100 - 100.0 * (l[0] * l[1] + l[1] * l[2] + l[2] * 10.) / (784. * 300. + 300.*100. + 100.*10.)
        return pruning_rate

    def z_phis(self):
        return [layer.z_phi for layer in self.layers]

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

    def set_mask_threshold(self):
        sum_pi = torch.zeros([0], dtype=torch.float32)
        for layer in self.layers:
            sum_pi = torch.cat([sum_pi, layer.sum_pi.cpu()], 0)
        threshold = np.percentile(sum_pi.data.numpy(), self.opt.pruningrate)
        for layer in self.layers:
            layer.mask_threshold = threshold
        print('th', threshold)


