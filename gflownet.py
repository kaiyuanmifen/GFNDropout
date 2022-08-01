import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FlowFunc(nn.Module):
    def __init__(
            self,
            state_dim,
            condition_dim,
            n_action,
            embedding_dim,
            hidden,
            activation=nn.LeakyReLU
    ):
        super().__init__()
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.n_action = n_action
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.n_action, self.embedding_dim)
        self.activation = activation

        h_old = self.embedding_dim + state_dim + condition_dim
        self.fc = nn.ModuleList()
        for h in hidden:
            self.fc.append(nn.Linear(h_old, h))
            h_old = h
        self.out_layer = nn.Linear(h_old, 1)

    def forward(self, state, action, condition=None):
        """

        :param state:
        :param action: one-hot
        :param condition:
        :return:
        """
        embed = self.embedding(torch.argmax(action, 1))
        if condition is None:
            x = torch.cat([state, embed], 1)
        else:
            x = torch.cat([state, embed, condition], 1)
        for layer in self.fc:
            x = self.activation()(layer(x))
        return self.out_layer(x)


class GFlowNet(object):
    """
    GFlowNet
    """

    def __init__(
            self,
            N,
            max_step,
            flow_func,
            device,
            beta=1.0,
            gamma=0.1,
            eps=1e-8,
    ):
        self.N = N
        self.max_step = max_step
        self.flow_func = flow_func
        self.state_dim = flow_func.state_dim
        self.n_action = flow_func.n_action
        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.eps = torch.tensor(eps).type(torch.float32)

        self.action_indice = torch.arange(self.N).to(self.device)
        self.action_space = F.one_hot(self.action_indice.flatten()).to(self.device)

    def get_flow_out(self, state, condition=None):
        pool_size = int(state[0].sum().item())
        batch_size = state.shape[0]
        possible_actions = self.action_space.repeat(batch_size, 1)[(state == 1).reshape((-1,))].reshape(
            (-1, self.n_action))
        states = state.repeat(1, pool_size).reshape((-1, state.shape[-1]))
        conditions = condition.repeat(1, pool_size).reshape(
            (-1, condition.shape[-1])) if condition is not None else None
        log_flows = self.flow_func(states, possible_actions, conditions)

        return possible_actions, log_flows

    def get_flow_in(self, state, condition=None):
        pool_size = self.N - int(state[0].sum().item())
        batch_size = state.shape[0]
        possible_backward_actions = self.action_space.repeat(batch_size, 1)[(state == 0).reshape((-1,))].reshape(
            (-1, self.n_action))
        parent_states = state.repeat(1, pool_size).reshape((-1, state.shape[-1]))
        conditions = condition.repeat(1, pool_size).reshape(
            (-1, condition.shape[-1])) if condition is not None else None
        log_flows = self.flow_func(parent_states, possible_backward_actions, conditions)

        return possible_backward_actions, log_flows

    def sample(self, n_traj, condition=None):
        trajs = []
        states = torch.ones(n_traj, self.N).to(self.device)
        for step in range(self.max_step):
            pool_size = self.N - step
            possible_actions, log_flows = self.get_flow_out(states, condition)
            flows = torch.exp(log_flows).reshape((n_traj, -1))
            probs = torch.nn.Softmax(1)(self.beta * flows).detach().cpu().numpy()
            action_indice = []
            for k in range(n_traj):
                if np.random.uniform() < self.gamma:
                    idx = np.random.choice(np.arange(len(probs[k])))
                else:
                    idx = np.random.choice(np.arange(len(probs[k])), p=probs[k])
                action_idx = int(torch.argmax(possible_actions[k * pool_size + idx]).item())
                action_indice.append(action_idx)
            trajs.append(action_indice)
            actions = self.action_space[action_indice]
            states = states - actions

        return np.array(trajs).T, states

    def loss(self, trajs, rewards, condition=None):
        n_traj = len(trajs)
        states = torch.zeros(n_traj, self.state_dim).to(self.device)
        loss = torch.tensor(0).type(torch.float32)
        for step in range(self.max_step):
            if step > 0:
                _, log_flows_in = self.get_flow_in(states, condition)
                if step == self.N - 1:
                    r = rewards
                    log_flows_out = torch.zeros(rewards.shape).unsqueeze(1)
                else:
                    r = torch.zeros(rewards.shape)
                    _, log_flows_out = self.get_flow_out(states, condition)
                in_log_sum = torch.logsumexp(
                    torch.cat([
                        (torch.log(self.eps) + torch.zeros(rewards.shape)).unsqueeze(1),
                        log_flows_in.reshape((n_traj, -1))
                    ], dim=1),
                    dim=1)
                out_log_sum = torch.logsumexp(
                    torch.cat([
                        (torch.log(self.eps) + torch.log(r)).unsqueeze(1),
                        log_flows_out.reshape((n_traj, -1))
                    ], dim=1),
                    dim=1)
                loss += torch.mean((in_log_sum - out_log_sum) ** 2)
            action_indice = trajs[:, step]
            actions = self.action_space[action_indice]
            states = states - actions
        return loss


if __name__ == '__main__':
    N = 5
    flow_func = FlowFunc(
        state_dim=N,
        condition_dim=0,
        n_action=N,
        embedding_dim=16,
        hidden=[32, 32],
        activation=nn.LeakyReLU
    )
    gfn = GFlowNet(
        N=N,
        max_step=N,
        flow_func=flow_func,
        device='cpu',
        beta=1.0,
        eps=0.1,
    )
    print(gfn.sample(3))
