#resnet18 code for contexual and concrete dropout
from models.layer import ArmConv2d
from models.layer.MAPConv2D import MAPConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from models.layer.MAPDense import MAPDense

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        self.to(device)
    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
            #x = F.relu (x)
        else:
            out = F.relu(self.bn1(x))
            #out = F.relu(x)
        out = self.conv1(out if self.equalInOut else x)
        out = self.conv2(F.relu(self.bn2(out)))
        #out = self.conv2(F.relu(out))
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01,
                 local_rep=False,opt=None):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init,
                                      weight_decay=weight_decay, lamba=lamba, local_rep=local_rep,opt=opt)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init,
                    weight_decay=0., lamba=0.01, local_rep=False,opt=None):
        self.layers = []
        for i in range(nb_layers):
            self.layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1,
                                droprate_init, weight_decay, lamba, local_rep,opt))
        return nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layer(x)


class ResNet_Con(nn.Module):
    # droprate_init = 0.3
    def __init__(self, depth=14, num_classes=10, widen_factor=1, N=50000, beta_ema=0.99, weight_decay=5e-4,
                 lambas=0.001,opt=None):
        super(ResNet_Con, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt=opt

        nChannels = [64, 128, 256, 512]
        #assert ((depth - 4) % 6 == 0)
        #self.n = (depth - 4) // 6 # 4
        self.n = 2
        
        self.N = N
        self.beta_ema = beta_ema
        self.epoch = 0
        self.elbo = 0
        block = BasicBlock
        droprate_init = self.opt.wrn_dr

        self.weight_decay = N * weight_decay
        self.lamba = lambas

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)
        # 1st block
        self.block1 = NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=self.opt.local_rep,opt=self.opt)
        # 2nd block
        self.block2 = NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=self.opt.local_rep,opt=self.opt)
        # 3rd block
        self.block3 = NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=self.opt.local_rep,opt=self.opt)
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

        self.to(self.device)
        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))


        
        

        self.to(self.device)

    def score(self, x, y=None):
        x=x.to(self.device) 
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        #out = F.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    def score2(self, x, y=None):
        x=x.to(self.device)
        out = self.conv1(x)
        # block1
        for i in range(4):
            x=out
            if not self.block1.layers[i].equalInOut:
                self.train() if self.opt.gpus <= 1 else self.module.train()
                x = F.relu(self.block1.layers[i].bn1(x))
            else:
                out = F.relu(self.block1.layers[i].bn1(x))
            self.eval() if self.opt.gpus <= 1 else self.module.eval()
            out = self.block1.layers[i].conv1(out if self.block1.layers[i].equalInOut else x)
            self.train() if self.opt.gpus <= 1 else self.module.train()
            out = self.block1.layers[i].bn2(out)
            self.eval() if self.opt.gpus <= 1 else self.module.eval()
            out = self.block1.layers[i].conv2(F.relu(out))
            out = torch.add(out, x if self.block1.layers[i].equalInOut else self.block1.layers[i].convShortcut(x))
        # block2
        for i in range(4):
            x = out
            if not self.block2.layers[i].equalInOut:
                self.train() if self.opt.gpus <= 1 else self.module.train()
                x = F.relu(self.block2.layers[i].bn1(x))
            else:
                out = F.relu(self.block2.layers[i].bn1(x))
            self.eval() if self.opt.gpus <= 1 else self.module.eval()
            out = self.block2.layers[i].conv1(out if self.block2.layers[i].equalInOut else x)
            self.train() if self.opt.gpus <= 1 else self.module.train()
            out = self.block2.layers[i].bn2(out)
            self.eval() if self.opt.gpus <= 1 else self.module.eval()
            out = self.block2.layers[i].conv2(F.relu(out))
            out = torch.add(out, x if self.block2.layers[i].equalInOut else self.block2.layers[i].convShortcut(x))
        # block3
        for i in range(4):
            x = out
            if not self.block3.layers[i].equalInOut:
                self.train() if self.opt.gpus <= 1 else self.module.train()
                x = F.relu(self.block3.layers[i].bn1(x))
            else:
                out = F.relu(self.block3.layers[i].bn1(x))
            self.eval() if self.opt.gpus <= 1 else self.module.eval()
            out = self.block3.layers[i].conv1(out if self.block3.layers[i].equalInOut else x)
            self.train() if self.opt.gpus <= 1 else self.module.train()
            out = self.block3.layers[i].bn2(out)
            self.eval() if self.opt.gpus <= 1 else self.module.eval()
            out = self.block3.layers[i].conv2(F.relu(out))
            out = torch.add(out, x if self.block3.layers[i].equalInOut else self.block3.layers[i].convShortcut(x))
        self.train() if self.opt.gpus <= 1 else self.module.train()
        out = F.relu(self.bn(out))
        self.eval() if self.opt.gpus <= 1 else self.module.eval()
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)




    def update_phi_gradient(self, f1, f2, update_flag=None):
        #flag
        if self.opt.dptype:
            if not self.opt.se:
                if update_flag is not None:
                    for i, layer in enumerate(self.block_list):
                        if update_flag[i]:
                            layer.conv1.update_phi_gradient(f1[i], f2)
                else:
                    for i, layer in enumerate(self.block_list):
                        layer.conv1.update_phi_gradient(f1[i], f2)
        else:
            for layer in self.l0_layers:
                layer.update_phi_gradient(f1, f2)

    def forward_mode(self, mode):
        #flag
        if self.opt.dptype:
            for i, layer in enumerate(self.l0_layers):
                layer.forward_mode = mode[i]
        else:
            for layer in self.l0_layers:
                layer.forward_mode = mode

    def forward(self, x, y=None):
        #flag:
        x=x.to(self.device)
        self.block_list = self.block1.layers + self.block2.layers + self.block3.layers
        if self.opt.var_dropout:
            if self.training:
                score = self.score(x)
                if self.epoch <= self.opt.N_t:
                    beta = (1.0/self.opt.N_t) * self.epoch
                else:
                    beta = 1
                kl = 0
                for i in range(len(self.block_list)):
                    kl = kl + self.block_list[i].conv1.get_kl()
                kl_loss = (1.0/self.N) * (self.opt.lambda_kl * beta * kl)
                kl_loss.backward(retain_graph = True)
            else:
                score = self.score(x)
                self.elbo = self.elbo_fun(nn.CrossEntropyLoss(reduce=False)(score, y).data)
            return score
        elif self.opt.dptype:
            if self.opt.se:
                if self.opt.batchtrain:
                    self.forward_mode([True] * len(self.l0_layers))
                    score = self.score2(x)
                else:
                    self.forward_mode([True] * len(self.l0_layers))
                    score = self.score(x)
                return score
            else:
                if self.training:
                    if self.opt.optim_method:
                        if self.opt.batchtrain:
                            self.forward_mode([True] * len(self.l0_layers))
                            score = self.score2(x)
                        else:
                            if self.opt.lambda_kl != 0.0:
                                f_kl = 0
                                f_prior = 0
                                self.forward_mode([True] * len(self.l0_layers))
                                score = self.score(x)
                                for i in range(len(self.block_list)):
                                    f_kl = f_kl + self.block_list[i].conv1.post_nll_true - self.block_list[i].conv1.prior_nll_true
                                    f_prior = f_prior + self.block_list[i].conv1.prior_nll_true
                                # if self.opt.learn_prior:
                                #     f_prior.mean().backward(retain_graph=True)
                                kl_loss = (- self.opt.lambda_kl * f_kl).mean()
                                kl_loss.backward(retain_graph=True)
                                # print('grad', self.block_list[1].conv1.eta.grad)
                            else:
                                self.forward_mode([True] * len(self.l0_layers))
                                score = self.score(x)
                        return score
                    else:
                        if self.opt.ctype =="Gaussian":
                            f_kl = 0
                            #f_prior = 0
                            self.forward_mode([True] * len(self.l0_layers))
                            score = self.score(x)
                            for i in range(len(self.block_list)):
                                f_kl = f_kl + self.block_list[i].conv1.post_nll_true - self.block_list[i].conv1.prior_nll_true
                                #f_prior = f_prior + self.block_list[i].conv1.prior_nll_true
                            # if self.opt.learn_prior:
                            #     f_prior.mean().backward(retain_graph=True)
                            kl_loss = (- self.opt.lambda_kl * f_kl).mean()
                            #if self.opt.learn_prior:
                            #    f_prior.mean().backward(retain_graph = True)
                            kl_loss.backward(retain_graph=True)
                            # print('grad', self.block_list[1].conv1.eta.grad)
                        else:
                            out = self.conv1(x)
                            f1app = []
                            f2_kl = 0
                            f2_prior = 0
                            update_flag = []
                            for i in range(len(self.block_list)):
                                # true actions
                                self.forward_mode([True] * len(self.l0_layers))
                         
                                main_traj = self.block_list[i].to(self.device)(out)
                                f1_kl = f2_kl
                                f2_kl = f2_kl + self.block_list[i].conv1.post_nll_true - self.block_list[i].conv1.prior_nll_true
                                f2_prior = f2_prior + self.block_list[i].conv1.prior_nll_true
                                # pseudo actions
                                # TODO: make sure u stays the same, and always sample instead of greedy, the layers are matching.
                                self.forward_mode([False] * len(self.l0_layers))
                                pseudo_traj = self.block_list[i](out).clone()
                                f1_kl = f1_kl + self.block_list[i].conv1.post_nll_sudo - self.block_list[i].conv1.prior_nll_sudo
                                if self.block_list[i].conv1.new_pseudo:
                                    self.forward_mode([True] * len(self.l0_layers))
                                    for k in range(i+1, len(self.block_list)):
                                        pseudo_traj = self.block_list[k](pseudo_traj)
                                        f1_kl = f1_kl + self.block_list[k].conv1.post_nll_true - self.block_list[k].conv1.prior_nll_true
                                    pseudo_traj = F.relu(self.bn(pseudo_traj))
                                    pseudo_traj = F.avg_pool2d(pseudo_traj, 8)
                                    pseudo_traj = pseudo_traj.view(pseudo_traj.size(0), -1)
                                    pseudo_score = self.fcout(pseudo_traj).data
                                    f1 = nn.CrossEntropyLoss(reduce=False)(pseudo_score, y).data - self.opt.lambda_kl * f1_kl.data
                                    f1 = f1 / f1.size(0)
                                    f1app.append(f1)
                                    update_flag.append(True)
                                else:
                                    f1app.append(0.0)
                                    update_flag.append(False)
                                    # TODO: change update phi.
                                out = main_traj
                            out = F.relu(self.bn(out))
                            out = F.avg_pool2d(out, 8)
                            out = out.view(out.size(0), -1)
                            score = self.fcout(out)
                            f2 = nn.CrossEntropyLoss(reduce=False)(score.data, y).data - self.opt.lambda_kl * f2_kl.data
                            f2 = f2 / f2.size(0)
                            self.update_phi_gradient(f1app, f2, update_flag)
                            if self.opt.learn_prior:
                                f2_prior.mean().backward(retain_graph=True)
                else:
                    if self.opt.batchtrain:
                        self.forward_mode([True] * len(self.l0_layers))
                        score = self.score2(x)

                    else:
                        self.forward_mode([True] * len(self.l0_layers))
                        score = self.score(x)
                    self.elbo = self.elbo_fun(nn.CrossEntropyLoss(reduce=False)(score, y).data)
                return score

        else:
            if self.opt.concretedp:
                if self.opt.gumbelconcrete:
                    if self.training:
                        if self.opt.lambda_kl != 0.0:
                            self.block_list = self.block1.layers + self.block2.layers + self.block3.layers
                            f_kl = 0
                            f_prior = 0
                            self.forward_mode(True)
                            score = self.score(x)
                            for i in range(len(self.block_list)):
                                f_kl = f_kl + self.block_list[i].conv1.post_nll_true - self.block_list[i].conv1.prior_nll_true
                                f_prior = f_prior + self.block_list[i].conv1.prior_nll_true
                            # if self.opt.learn_prior:
                            #     f_prior.mean().backward(retain_graph=True)
                            kl_loss = (- self.opt.lambda_kl * (1.0 / 60000.0)* f_kl).mean()
                            kl_loss.backward(retain_graph=True)
                            # print('grad', self.block_list[1].conv1.eta.grad)
                    else:
                        self.forward_mode(True)
                        score = self.score(x)
                        self.elbo = self.elbo_fun(nn.CrossEntropyLoss(reduce=False)(score, y).data)
                else:
                    if self.training:
                        self.forward_mode(True)
                        score = self.score(x)
                        self.block_list = self.block1.layers + self.block2.layers + self.block3.layers
                        f1_kl = 0
                        f2_kl = 0
                        f1_prior = 0
                        f2_prior = 0
                        for i in range(len(self.block_list)):
                            # true actions
                            self.forward_mode(True)
                            f2_kl = f2_kl + self.block_list[i].conv1.post_nll_true - self.block_list[i].conv1.prior_nll_true
                            f2_prior = f2_prior + self.block_list[i].conv1.prior_nll_true
                        self.eval() if self.opt.gpus <= 1 else self.module.eval()
                        if self.opt.ar is not True:
                            self.forward_mode(False)
                            score2 = self.score(x).data
                            f1 = nn.CrossEntropyLoss(reduce=False)(score2, y).data
                            for i in range(len(self.block_list)):
                                f1_kl = f1_kl + self.block_list[i].conv1.post_nll_sudo - self.block_list[i].conv1.prior_nll_sudo
                                f1_prior = f1_prior + self.block_list[i].conv1.prior_nll_sudo
                        else:
                            f1 = 0
                        f2 = nn.CrossEntropyLoss(reduce=False)(score, y).data- self.opt.lambda_kl * (1.0 / 60000.0)* f2_kl.data
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
                    if self.opt.batchtrain:
                        self.forward_mode(True)
                        score = self.score2(x)

                    else:
                        self.forward_mode(True)
                        score = self.score(x)
                    self.elbo = -nn.CrossEntropyLoss(reduce=False)(score, y).data.mean()
            return score

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.weight_decay > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
            except:
                pass
        return expected_flops, expected_l0

    #Updating:
    def get_dprate(self):
        # flag
        if self.opt.dptype:
            dprate = []
            i=0
            for layer in self.l0_layers:
                if i >= 1:
                    break
                i += 1
                #dprate.append(torch.mean(layer.pi).cpu().item())
                dprate.append((layer.pi.detach()).cpu().numpy())
            return dprate
        else:
            return 0.0

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
        return [layer.activated_neurons() for layer in self.l0_layers]

    def get_expected_activated_neurons(self):
        return [layer.expected_activated_neurons() for layer in self.l0_layers]

    def prune_rate(self):
        if self.opt.var_dropout:
            return 0
        l = [layer.activated_neurons().cpu().numpy() for layer in self.l0_layers]
        if self.opt.dptype:
            pruning_rate = 100 - 100. * ((160**2 *4 + 320**2*4+640**2*4)/self.opt.cha_factor + 160 * 16 + (l[1] + l[2] + l[3] + l[0]) * 160 + (l[5] + l[6] + l[7] + l[4]) * 320 + (
                    l[9] + l[10] + l[8]) * 640) \
               / (16 * 160 + 160 * 160 * 3 + 160 * 320 + 320 * 320 * 3 + 320 * 640 + 640 * 640 * 3)
            pruning_rate_2 = 100 - 100. * (160 * 16 + (l[1] + l[2] + l[3] + l[0]) * 160 + (l[5] + l[6] + l[7] + l[4]) * 320 + (
                    l[9] + l[10] + l[8]) * 640) \
               / (16 * 160 + 160 * 160 * 3 + 160 * 320 + 320 * 320 * 3 + 320 * 640 + 640 * 640 * 3)
            print('decoder pruning rate', pruning_rate_2)
        else:
            pruning_rate = 100 - 100. * (l[0] * 16 + (l[1] + l[2] + l[3] + l[4]) * 160 + (l[5] + l[6] + l[7] + l[8]) * 320 + (
                    l[9] + l[10] + l[11]) * 640) \
               / (16 * 160 + 160 * 160 * 3 + 160 * 320 + 320 * 320 * 3 + 320 * 640 + 640 * 640 * 3)
        return pruning_rate

    def z_phis(self):
        return [layer.z_phi for layer in self.l0_layers]

    def elbo_fun(self, nn_loss):
        elbo = nn_loss
        if self.opt.var_dropout:
            elbo = elbo.mean()
            for i in range(len(self.block_list)):
                elbo = elbo + (1/self.N) * self.block_list[i].conv1.get_kl().detach()
            return -elbo
        for i in range(len(self.block_list)):
            # true actions
            ## Add this
            if self.opt.dptype:
                elbo = elbo - (self.block_list[i].conv1.post_nll_true - self.block_list[i].conv1.prior_nll_true).data
            else:
                elbo = elbo - (1.0 / 60000.0) * (self.block_list[i].conv1.post_nll_true - self.block_list[i].conv1.prior_nll_true).data
        return -elbo.mean()

