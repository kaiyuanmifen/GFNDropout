import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import zipfile


class Dropout_variants(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1, dp_type=False, concretedp=True,
                 learn_prior=False, out_features=1,
                 k=1, eta_const=-1.38, cha_factor=1):
        super(Dropout_variants, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.cha_factor = cha_factor
        self.dp_type = dp_type
        self.concretedp = concretedp
        self.learn_prior = learn_prior
        self.k = k
        self.use_bias = True

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        if dp_type:
            self.ctdo_linear2 = nn.Linear(out_features, int(out_features / self.cha_factor), bias=self.use_bias)
            self.ctdo_linear3 = nn.Linear(int(out_features / self.cha_factor), out_features, bias=self.use_bias)
            if self.use_bias:
                self.ctdo_linear3.bias.data.fill_(eta_const)
                self.ctdo_linear2.bias.data.fill_(eta_const)
                self.ctdo_linear3.weight.data.normal_(0, 1 / 10)
                self.ctdo_linear2.weight.data.normal_(0, 1 / 10)
            if learn_prior:
                self.eta = nn.Parameter(torch.Tensor(1))
                self.eta.data.fill_(eta_const)
            else:
                self.eta = torch.from_numpy(np.ones(out_features)) * eta_const
        else:
            if concretedp:
                self.z_phi = nn.Parameter(torch.empty(1).data.fill_(eta_const))
                self.eta = torch.from_numpy(np.ones(out_features)) * eta_const
            else:
                # fix distribution
                self.z_phi = torch.from_numpy(np.ones([1]) * eta_const).type(torch.float32)

    def contextual_dropout(self, input):
        if self.dp_type:
            z_phi = self.ctdo_linear2(input.data)
            m = nn.LeakyReLU(0.3)
            z_phi = m(z_phi)
            z_phi = self.ctdo_linear3(z_phi)
            return z_phi

    def forward(self, x, layer):
        if self.dp_type:
            self.z_phi = self.contextual_dropout(x)
        pi = torch.sigmoid(self.k * self.z_phi)
        out = layer(self._concrete_dropout(x, pi))

        #         out = layer(x)
        #         if self.dp_type:
        #             self.z_phi = self.contextual_dropout(out)
        #         pi = torch.sigmoid(self.k * self.z_phi)
        #         out = self._concrete_dropout(out, pi)

        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - pi)

        dropout_regularizer = pi * torch.log(pi)
        dropout_regularizer += (1. - pi) * torch.log(1. - pi)
        input_dimensionality = x[0].numel()  # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        if self.dp_type:
            regularization = ((self.post_nll_true - self.prior_nll_true) * self.dropout_regularizer).mean()
            # print('regularization', regularization.shape, 'pripr_nll', self.prior_nll_true.shape)

        else:
            if self.concretedp:
                # regularization = ((self.post_nll_true - self.prior_nll_true) * self.dropout_regularizer).mean()
                regularization = weights_regularizer + dropout_regularizer
            else:
                regularization = 0
        return out, regularization

    def _concrete_dropout(self, x, pi):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(pi + eps)
                     - torch.log(1 - pi + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - pi

        if self.dp_type or self.concretedp:
            self.post_nll_true = - (drop_prob * torch.log(pi + eps) + (1 - drop_prob) * torch.log(1 - pi + eps))
            self.post_nll_true = self.post_nll_true.mean(1)
            prior_pi = torch.sigmoid(self.k * self.eta.unsqueeze(0)).type_as(self.post_nll_true)
            self.prior_nll_true = - (
                        drop_prob * torch.log(prior_pi + eps) + (1 - drop_prob) * torch.log(1 - prior_pi + eps))
            self.prior_nll_true = self.prior_nll_true.mean(1)
        x = torch.mul(x, random_tensor)
        # x /= retain_prob # think about this. no testing mode?

        return x


class Model(nn.Module):
    def __init__(self, nb_features, weight_regularizer, dropout_regularizer,
                 dp_type=False, concretedp=True, learn_prior=False,
                 k=1, eta_const=-1.38, in_features=1):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_features, nb_features)  # TODO: change dim as an input
        self.linear2 = nn.Linear(nb_features, nb_features)
        self.linear3 = nn.Linear(nb_features, nb_features)

        self.linear4_mu = nn.Linear(nb_features, 1)
        self.linear4_logvar = nn.Linear(nb_features, 1)

        self.conc_drop1 = Dropout_variants(weight_regularizer=weight_regularizer,
                                           dropout_regularizer=dropout_regularizer,
                                           dp_type=dp_type, concretedp=concretedp,
                                           learn_prior=learn_prior, out_features=in_features,
                                           k=k, eta_const=eta_const, cha_factor=0.2)
        self.conc_drop2 = Dropout_variants(weight_regularizer=weight_regularizer,
                                           dropout_regularizer=dropout_regularizer,
                                           dp_type=dp_type, concretedp=concretedp,
                                           learn_prior=learn_prior, out_features=nb_features,
                                           k=k, eta_const=eta_const, cha_factor=64)
        self.conc_drop3 = Dropout_variants(weight_regularizer=weight_regularizer,
                                           dropout_regularizer=dropout_regularizer,
                                           dp_type=dp_type, concretedp=concretedp,
                                           learn_prior=learn_prior, out_features=nb_features,
                                           k=k, eta_const=eta_const, cha_factor=64)
        self.conc_drop_mu = Dropout_variants(weight_regularizer=weight_regularizer,
                                             dropout_regularizer=dropout_regularizer,
                                             dp_type=dp_type, concretedp=concretedp,
                                             learn_prior=learn_prior, out_features=nb_features,
                                             k=k, eta_const=eta_const, cha_factor=64)
        self.conc_drop_logvar = Dropout_variants(weight_regularizer=weight_regularizer,
                                                 dropout_regularizer=dropout_regularizer,
                                                 dp_type=dp_type, concretedp=concretedp,
                                                 learn_prior=learn_prior, out_features=nb_features,
                                                 k=k, eta_const=eta_const, cha_factor=64)

        self.relu = nn.ReLU()

    def forward(self, x):
        regularization = torch.empty(5, device=x.device)

        #         x1, regularization[0] = self.conc_drop1(x, nn.Sequential(self.linear1, self.relu))
        x1 = nn.Sequential(self.linear1, self.relu)(x)
        regularization[0] = 0
        x2, regularization[1] = self.conc_drop2(x1, nn.Sequential(self.linear2, self.relu))
        x3, regularization[2] = self.conc_drop3(x2, nn.Sequential(self.linear3, self.relu))

        mean, regularization[3] = self.conc_drop_mu(x3, self.linear4_mu)
        log_var, regularization[4] = self.conc_drop_logvar(x3, self.linear4_logvar)

        #         mean = self.linear4_mu(x3)
        #         log_var = self.linear4_logvar(x3)

        return mean, log_var, regularization.sum()


def heteroscedastic_loss(true, mean, log_var):
    precision = torch.exp(-log_var)
    #     return torch.mean(torch.sum(precision * (true - mean)**2 + log_var, 1), 0)
    #     precision = torch.exp(-log_var)
    return torch.mean(torch.sum((true - mean) ** 2, 1), 0)


def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max


def test(Y_true, K_test, means, logvar):
    """
    Estimate predictive log likelihood:
    log p(y|x, D) = log int p(y|x, w) p(w|D) dw
                 ~= log int p(y|x, w) q(w) dw
                 ~= log 1/K sum p(y|x, w_k) with w_k sim q(w)
                  = LogSumExp log p(y|x, w_k) - log K
    :Y_true: a 2D array of size N x dim
    :MC_samples: a 3D array of size samples K x N x 2*D
    """
    k = K_test
    N = Y_true.shape[0]
    mean = means
    logvar = logvar
    test_ll = -0.5 * np.exp(-logvar) * (mean - Y_true.squeeze()) ** 2. - 0.5 * logvar - 0.5 * np.log(
        2 * np.pi)  # Y_true[None]
    test_ll = np.sum(np.sum(test_ll, -1), -1)
    test_ll = logsumexp(test_ll) - np.log(k)
    pppp = test_ll / N  # per point predictive probability
    rmse = np.mean((np.mean(mean, 0) - Y_true.squeeze()) ** 2.) ** 0.5
    return pppp, rmse


def QQ_plot(X_val, Y_val, means, quantile_num):
    # means shape: sample size, batch size
    # X_val shape: batch size,  1,
    quantiles = np.percentile(means, q=np.arange(0, 100, quantile_num), axis=0)
    # number of quantile, batch size
    data_quantile = np.mean(np.transpose(quantiles) > Y_val, 0)
    plt.figure()
    plt.plot(np.arange(0, 100, quantile_num) / 100.0, data_quantile, 'o')
    plt.plot(np.arange(0, 100, quantile_num) / 100.0, np.arange(0, 100, quantile_num) / 100.0)
    print('QQ mean difference: ', np.mean(np.abs(data_quantile - np.arange(0, 100, quantile_num) / 100.0)))

    plt.title('QQ-plot: uncertainty estimation')
    plt.show()
    return np.mean(np.abs(data_quantile - np.arange(0, 100, quantile_num) / 100.0))


def fit_model(nb_epoch, X, Y, nb_features, dp_type=False, concretedp=True, learn_prior=False,
              k=1, eta_const=-1.38, lr=0.01, in_features=1, batch_size=100):
    N = X.shape[0]
    wr = l ** 2. / N
    if concretedp:
        dr = 2. / N
    else:
        dr = 1.0
    model = Model(nb_features, wr, dr, dp_type=dp_type, concretedp=concretedp, learn_prior=learn_prior,
                  k=k, eta_const=eta_const, in_features=in_features)
    model = model  # .cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    perm_ind = np.random.permutation(np.arange(np.shape(X)[0]))
    X = X[perm_ind, :]
    Y = Y[perm_ind]
    for i in range(nb_epoch):
        old_batch = 0
        for batch in range(int(np.ceil(X.shape[0] / batch_size))):
            batch = (batch + 1)
            _x = X[old_batch: batch_size * batch]
            _y = Y[old_batch: batch_size * batch]

            x = Variable(torch.FloatTensor(_x))  # .cuda()
            y = Variable(torch.FloatTensor(_y))  # .cuda()

            mean, log_var, regularization = model(x)

            loss = heteroscedastic_loss(y, mean, log_var) + regularization
            loss_list.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i % 10 == 0:
            print('mean', torch.sigmoid(k * model.conc_drop2.z_phi).mean(), 'mean',
                  torch.sigmoid(k * model.conc_drop3.z_phi).mean(), 'loss', loss)
    plt.figure()
    plt.plot(loss_list)
    plt.title('loss')
    plt.show()
    return model


def plot(X_train, Y_train, X_val, Y_val, means):
    indx = np.argsort(X_val[:, 0])
    _, (ax1, ax2, ax3, ax4) = pylab.subplots(1, 4, figsize=(12, 1.5), sharex=True, sharey=True)
    ax1.scatter(X_train[:, 0], Y_train[:, 0], c='y')
    ax1.set_title('Train set')
    ax2.plot(X_val[indx, 0], np.mean(means, 0)[indx], color='skyblue', lw=3)
    ax2.scatter(X_train[:, 0], Y_train[:, 0], c='y')
    ax2.set_title('+Predictive mean')
    for mean in means:
        ax3.scatter(X_val[:, 0], mean, c='b', alpha=0.2, lw=0)
    ax3.plot(X_val[indx, 0], np.mean(means, 0)[indx], color='skyblue', lw=3)
    ax3.set_title('+MC samples on validation X')
    ax4.scatter(X_val[:, 0], Y_val[:, 0], c='r', alpha=0.2, lw=0)
    ax4.set_title('Validation set')
    pylab.show()


def gen_data(N):
    """
    Function to generate data
    """
    sigma = 0.05  # ground truth
    X = np.random.randn(N, Q)  # * 10
    w = 0.3
    b = 0.0
    Y = X * (X > -1).dot(w) + X * (X < -1).dot(-w) + b + sigma * np.random.randn(N, D) * (
                X < -1) + 2 * sigma * np.random.randn(N, D) * (X > -1)
    Y = X.dot(w) + b + sigma * np.random.randn(N, D) * (X < 0) + 1 * sigma * np.random.randn(N, D) * (
                X > 0) + 5 * np.random.binomial(p=0.5, n=1, size=[N, D]) * (X < 0.5) * (X > -0.5)

    return X, Y


import pylab
% matplotlib
inline

X, Y = gen_data(10)
pylab.figure(figsize=(3, 1.5))
pylab.scatter(X[:, 0], Y[:, 0], edgecolor='b')
pylab.show()

X, Y = gen_data(10000)
pylab.figure(figsize=(3, 1.5))
pylab.scatter(X[:, 0], Y[:, 0], edgecolor='b')
# pylab.xlim([-5, 5])
# pylab.ylim([-2, 20])
pylab.show()


N = 300
nb_epoch = 100
nb_val_size = 1000 # Validation size
nb_features = 64 # Hidden layer size
Q = 1 # Data dimensionality
D = 1 # One mean, one log_var
K_test = 100 # Number of MC samples
batch_size = 100
l = 1e-4 # Lengthscale
nb_reps = 5

################Concrete####################

seed = 10086
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
rep_results = []
k = 0.005
for i in range(nb_reps):
    X, Y = gen_data(N + nb_val_size)
    X_train, Y_train = X[:N], Y[:N]
    X_val, Y_val = X[N:], Y[N:]
    model = fit_model(nb_epoch, X_train, Y_train, nb_features, k=k, eta_const=0.0/k, lr=0.01)
    model.eval()
    MC_samples = [model(Variable(torch.FloatTensor(X_val))) for _ in range(K_test)]
    #MC_samples = [model(Variable(torch.FloatTensor(X_val)).cuda()) for _ in range(K_test)]
    means = torch.stack([tup[0] for tup in MC_samples]).view(K_test, X_val.shape[0]).cpu().data.numpy()
    logvar = torch.stack([tup[1] for tup in MC_samples]).view(K_test, X_val.shape[0]).cpu().data.numpy()
    pppp, rmse = test(Y_val, K_test, means, logvar)
    epistemic_uncertainty = np.var(means, 0).mean(0)
    logvar = np.mean(logvar, 0)
    aleatoric_uncertainty = np.exp(logvar).mean(0)
    ps = np.array([torch.sigmoid(k * module.z_phi).cpu().data.numpy()[0] for module in model.modules() if hasattr(module, 'z_phi')])
    plot(X_train, Y_train, X_val, Y_val, means)
    qq_score = QQ_plot(X_val, Y_val, means, 1)
    rep_results += [(rmse, ps, aleatoric_uncertainty, epistemic_uncertainty, pppp, qq_score)]
test_mean = np.mean([r[0] for r in rep_results])
test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(nb_reps)
test_nll_mean = np.mean([r[4] for r in rep_results])
test_nll_std_err = np.std([r[4] for r in rep_results]) / np.sqrt(nb_reps)
test_qq_mean = np.mean([r[5] for r in rep_results])
test_qq_std_err = np.std([r[5] for r in rep_results]) / np.sqrt(nb_reps)
ps = np.mean([r[1] for r in rep_results], 0)
aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
epistemic_uncertainty = np.mean([r[3] for r in rep_results])
print(N, nb_epoch, 'RMSE mean', test_mean, 'RMSE std',test_std_err,
      'QQ mean', test_qq_mean, 'QQ std',test_qq_std_err,'NLL mean:', test_nll_mean, 'NLL std:', test_nll_std_err,
      'dropout rate:',ps, ' - ', aleatoric_uncertainty**0.5, epistemic_uncertainty**0.5)

################FIX####################

seed = 10086
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
rep_results = []
k = 0.005
for i in range(nb_reps):
    X, Y = gen_data(N + nb_val_size)
    X_train, Y_train = X[:N], Y[:N]
    X_val, Y_val = X[N:], Y[N:]
    model = fit_model(nb_epoch, X_train, Y_train, nb_features, k=k, eta_const=0.0/k,
                      dp_type=False, concretedp=False, learn_prior=False, lr=0.01)
    model.eval()
    MC_samples = [model(Variable(torch.FloatTensor(X_val))) for _ in range(K_test)]
    #MC_samples = [model(Variable(torch.FloatTensor(X_val)).cuda()) for _ in range(K_test)]
    means = torch.stack([tup[0] for tup in MC_samples]).view(K_test, X_val.shape[0]).cpu().data.numpy()
    logvar = torch.stack([tup[1] for tup in MC_samples]).view(K_test, X_val.shape[0]).cpu().data.numpy()
    pppp, rmse = test(Y_val, K_test, means, logvar)
    epistemic_uncertainty = np.var(means, 0).mean(0)
    logvar = np.mean(logvar, 0)
    aleatoric_uncertainty = np.exp(logvar).mean(0)
    ps = np.array([torch.sigmoid(k * module.z_phi).cpu().mean().data.numpy() for module in model.modules() if hasattr(module, 'z_phi')])
    plot(X_train, Y_train, X_val, Y_val, means)
    qq_score = QQ_plot(X_val, Y_val, means, 1)
    rep_results += [(rmse, ps, aleatoric_uncertainty, epistemic_uncertainty, pppp, qq_score)]
test_mean = np.mean([r[0] for r in rep_results])
test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(nb_reps)
test_nll_mean = np.mean([r[4] for r in rep_results])
test_nll_std_err = np.std([r[4] for r in rep_results]) / np.sqrt(nb_reps)
test_qq_mean = np.mean([r[5] for r in rep_results])
test_qq_std_err = np.std([r[5] for r in rep_results]) / np.sqrt(nb_reps)
ps = np.mean([r[1] for r in rep_results], 0)
aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
epistemic_uncertainty = np.mean([r[3] for r in rep_results])
print(N, nb_epoch, 'RMSE mean', test_mean, 'RMSE std',test_std_err,
      'QQ mean', test_qq_mean, 'QQ std',test_qq_std_err,'NLL mean:', test_nll_mean, 'NLL std:', test_nll_std_err,
      'dropout rate:',ps, ' - ', aleatoric_uncertainty**0.5, epistemic_uncertainty**0.5)

################RBNN-non-LEARN-PRIOR####################


seed = 10086
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
rep_results = []
k = 0.005
for i in range(nb_reps):
    X, Y = gen_data(N + nb_val_size)
    X_train, Y_train = X[:N], Y[:N]
    X_val, Y_val = X[N:], Y[N:]
    model = fit_model(nb_epoch, X_train, Y_train, nb_features, k=k, eta_const=0.0/k, dp_type=True, concretedp=False, learn_prior=False, lr=0.01)
    model.eval()
    MC_samples = [model(Variable(torch.FloatTensor(X_val))) for _ in range(K_test)]
    #MC_samples = [model(Variable(torch.FloatTensor(X_val)).cuda()) for _ in range(K_test)]
    means = torch.stack([tup[0] for tup in MC_samples]).view(K_test, X_val.shape[0]).cpu().data.numpy()
    logvar = torch.stack([tup[1] for tup in MC_samples]).view(K_test, X_val.shape[0]).cpu().data.numpy()
    pppp, rmse = test(Y_val, K_test, means, logvar)
    epistemic_uncertainty = np.var(means, 0).mean(0)
    logvar = np.mean(logvar, 0)
    aleatoric_uncertainty = np.exp(logvar).mean(0)
    ps = np.array([torch.sigmoid(k * module.z_phi).cpu().mean().data.numpy() for module in model.modules() if hasattr(module, 'z_phi')])
    plot(X_train, Y_train, X_val, Y_val, means)
    qq_score = QQ_plot(X_val, Y_val, means, 1)
    rep_results += [(rmse, ps, aleatoric_uncertainty, epistemic_uncertainty, pppp, qq_score)]
test_mean = np.mean([r[0] for r in rep_results])
test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(nb_reps)
test_nll_mean = np.mean([r[4] for r in rep_results])
test_nll_std_err = np.std([r[4] for r in rep_results]) / np.sqrt(nb_reps)
test_qq_mean = np.mean([r[5] for r in rep_results])
test_qq_std_err = np.std([r[5] for r in rep_results]) / np.sqrt(nb_reps)
ps = np.mean([r[1] for r in rep_results], 0)
aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
epistemic_uncertainty = np.mean([r[3] for r in rep_results])
print(N, nb_epoch, 'RMSE mean', test_mean, 'RMSE std',test_std_err,
      'QQ mean', test_qq_mean, 'QQ std',test_qq_std_err,'NLL mean:', test_nll_mean, 'NLL std:', test_nll_std_err,
      'dropout rate:',ps, ' - ', aleatoric_uncertainty**0.5, epistemic_uncertainty**0.5)




################RBNN-LEARN-PRIOR####################

seed = 10086
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
rep_results = []
k = 0.005
for i in range(nb_reps):
    X, Y = gen_data(N + nb_val_size)
    X_train, Y_train = X[:N], Y[:N]
    X_val, Y_val = X[N:], Y[N:]
    model = fit_model(nb_epoch, X_train, Y_train, nb_features, k=k, eta_const=0.0/k, dp_type=True,
                      concretedp=False, learn_prior=True, lr=0.01)
    model.eval()
    MC_samples = [model(Variable(torch.FloatTensor(X_val))) for _ in range(K_test)]
    #MC_samples = [model(Variable(torch.FloatTensor(X_val)).cuda()) for _ in range(K_test)]
    means = torch.stack([tup[0] for tup in MC_samples]).view(K_test, X_val.shape[0]).cpu().data.numpy()
    logvar = torch.stack([tup[1] for tup in MC_samples]).view(K_test, X_val.shape[0]).cpu().data.numpy()
    pppp, rmse = test(Y_val, K_test, means, logvar)
    epistemic_uncertainty = np.var(means, 0).mean(0)
    logvar = np.mean(logvar, 0)
    aleatoric_uncertainty = np.exp(logvar).mean(0)
    ps = np.array([torch.sigmoid(k * module.z_phi).mean().cpu().data.numpy() for module in model.modules() if hasattr(module, 'z_phi')])
    plot(X_train, Y_train, X_val, Y_val, means)
    qq_score = QQ_plot(X_val, Y_val, means, 1)
    rep_results += [(rmse, ps, aleatoric_uncertainty, epistemic_uncertainty, pppp, qq_score)]
test_mean = np.mean([r[0] for r in rep_results])
test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(nb_reps)
test_nll_mean = np.mean([r[4] for r in rep_results])
test_nll_std_err = np.std([r[4] for r in rep_results]) / np.sqrt(nb_reps)
test_qq_mean = np.mean([r[5] for r in rep_results])
test_qq_std_err = np.std([r[5] for r in rep_results]) / np.sqrt(nb_reps)
ps = np.mean([r[1] for r in rep_results], 0)
aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
epistemic_uncertainty = np.mean([r[3] for r in rep_results])
print(N, nb_epoch, 'RMSE mean', test_mean, 'RMSE std',test_std_err,
      'QQ mean', test_qq_mean, 'QQ std',test_qq_std_err,'NLL mean:', test_nll_mean, 'NLL std:', test_nll_std_err,
      'dropout rate:',ps, ' - ', aleatoric_uncertainty**0.5, epistemic_uncertainty**0.5)


################UCI####################


def dropout_training_UCI(data, n_splits, nb_epoch, nb_features, dp_type=False,
                         concretedp=True, learn_prior=False, k=1, eta_const=-1.38, lr=0.01, K_test=100,
                         batch_size=None):
    kf = KFold(n_splits=n_splits)
    in_dim = data.shape[1] - 1
    rep_results = []
    for j, idx in enumerate(kf.split(data)):
        print('FOLD %d:' % j)
        train_index, test_index = idx
        if batch_size is None:
            batch_size = len(train_index)
        print(batch_size)
        x_train, y_train = data[train_index, :in_dim], data[train_index, in_dim:]
        x_test, y_test = data[test_index, :in_dim], data[test_index, in_dim:]
        x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0) ** 0.5
        y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0) ** 0.5

        x_train = (x_train - x_means) / x_stds
        y_train = (y_train - y_means) / y_stds

        x_test = (x_test - x_means) / x_stds
        y_test = (y_test - y_means) / y_stds
        model = fit_model(nb_epoch, x_train, y_train, nb_features, k=k, eta_const=eta_const,
                          dp_type=dp_type, concretedp=concretedp, learn_prior=learn_prior,
                          lr=lr, in_features=in_dim, batch_size=batch_size)
        model.eval()
        # MC_samples = [model(Variable(torch.FloatTensor(x_test))) for _ in range(K_test)]
        MC_samples = []
        for kk in range(K_test):
            MC_samples.append(model(Variable(torch.FloatTensor(x_test))))
            print(kk)

        #         MC_samples = [model((torch.from_numpy(x_test).type(torch.float32))) for _ in range(K_test)]
        means = torch.stack([tup[0] for tup in MC_samples]).view(K_test, x_test.shape[0]).cpu().data.numpy()
        logvar = torch.stack([tup[1] for tup in MC_samples]).view(K_test, x_test.shape[0]).cpu().data.numpy()
        pppp, rmse = test(y_test, K_test, means, logvar)
        epistemic_uncertainty = np.var(means, 0).mean(0)
        logvar = np.mean(logvar, 0)
        aleatoric_uncertainty = np.exp(logvar).mean(0)
        ps = np.array([torch.sigmoid(k * module.z_phi).cpu().mean().data.numpy() for module in model.modules() if
                       hasattr(module, 'z_phi')])
        # plot(x_train, y_train, X_val, Y_val, means)
        qq_score = QQ_plot(x_test, y_test, means, 1)
        rep_results += [(rmse, ps, aleatoric_uncertainty, epistemic_uncertainty, pppp, qq_score)]
    test_mean = np.mean([r[0] for r in rep_results])
    test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(n_splits)
    test_nll_mean = np.mean([r[4] for r in rep_results])
    test_nll_std_err = np.std([r[4] for r in rep_results]) / np.sqrt(n_splits)
    test_qq_mean = np.mean([r[5] for r in rep_results])
    test_qq_std_err = np.std([r[5] for r in rep_results]) / np.sqrt(n_splits)
    ps = np.mean([r[1] for r in rep_results], 0)
    aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
    epistemic_uncertainty = np.mean([r[3] for r in rep_results])
    print(N, nb_epoch, 'RMSE mean', test_mean, 'RMSE std', test_std_err,
          'QQ mean', test_qq_mean, 'QQ std', test_qq_std_err, 'NLL mean:', test_nll_mean, 'NLL std:', test_nll_std_err,
          'dropout rate:', ps, ' - ', aleatoric_uncertainty ** 0.5, epistemic_uncertainty ** 0.5)




################Data####################

# ############
# data_name = 'Boston'
# np.random.seed(0)
# #!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data" --no-check-certificate
# data = pd.read_csv('housing.data', header=0, delimiter="\s+").values
# data = data[np.random.permutation(np.arange(len(data)))]


# ############
# data_name = 'Concrete'
# np.random.seed(0)
# #!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls" --no-check-certificate
# data = pd.read_excel('Concrete_Data.xls', header=0, delimiter="\s+").values
# data = data[np.random.permutation(np.arange(len(data)))]


# ############
# data_name = 'Energy'
# np.random.seed(0)
# #!wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx" --no-check-certificate
# data = pd.read_excel('ENB2012_data.xlsx', header=0, delimiter="\s+").values
# data = data[np.random.permutation(np.arange(len(data)))]


############
# data_name = 'Power'
# np.random.seed(0)
# #!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip" --no-check-certificate
# zipped = zipfile.ZipFile("CCPP.zip")
# data = pd.read_excel(zipped.open('CCPP/Folds5x2_pp.xlsx'), header=0, delimiter="\t").values
# np.random.shuffle(data)

# ############
# data_name = 'Wine'
# np.random.seed(0)
# #!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" --no-check-certificate
# data = pd.read_csv('winequality-red.csv', header=1, delimiter=';').values
# data = data[np.random.permutation(np.arange(len(data)))]
# print(data.shape, data[:, -1].var()**0.5)


# ############
# data_name = 'Yacht'
# np.random.seed(0)
# #!wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data" --no-check-certificate
# data = pd.read_csv('yacht_hydrodynamics.data', header=1, delimiter='\s+').values
# data = data[np.random.permutation(np.arange(len(data)))]


# ############
# data_name = 'kin8nm'
# np.random.seed(0)
# # zipped = zipfile.ZipFile("UCI CBM Dataset.zip")
# data = pd.read_table('kin8nm.txt', delim_whitespace=True).values
# np.random.shuffle(data)
# # a = data[:, -1]
# # data[:, -2] = a
# # data = data[:, :-1] # predicting GT compressor decay state coeffcient



# ############
data_name = 'protein'
np.random.seed(0)
# zipped = zipfile.ZipFile("UCI CBM Dataset.zip")
data = pd.read_table('protein.txt', delim_whitespace=True).values
np.random.shuffle(data)




# data_name = 'Naval'
# np.random.seed(0)
# #!wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip" --no-check-certificate
# zipped = zipfile.ZipFile("UCI CBM Dataset.zip")
# data = pd.read_table(zipped.open('UCI CBM Dataset/data.txt'), delim_whitespace=True).values
# np.random.shuffle(data)
# # a = data[:, -1]
# # data[:, -2] = a
# # data = data[:, :-1] # predicting GT compressor decay state coeffcient



################RBNN-LEARN-PRIOR####################
seed = 10086
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

k=0.001
dropout_training_UCI(data, n_splits=2, nb_epoch=100, nb_features=64, dp_type=True,
                     concretedp=False, learn_prior=True, k=k, eta_const=0.0/k, lr=0.001, K_test=100, batch_size =100)



################Concrete####################

seed = 10086
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

k=0.005
dropout_training_UCI(data, n_splits=10, nb_epoch=100, nb_features=64, dp_type=False,
                     concretedp=True, learn_prior=False, k=k, eta_const=0.0/k, lr=0.01, K_test=100)



################FIX####################
seed = 10086
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

k=0.005
dropout_training_UCI(data, n_splits=2, nb_epoch=10, nb_features=64, dp_type=False,
                     concretedp=False, learn_prior=False, k=k, eta_const=0.0/k, lr=0.01, K_test=100, batch_size=100)



################RBNN-LEARN-PRIOR####################

seed = 10086
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

k=0.005
dropout_training_UCI(data, n_splits=10, nb_epoch=100, nb_features=64, dp_type=True,
                     concretedp=False, learn_prior=False, k=k, eta_const=0.0/k, lr=0.01, K_test=100)






