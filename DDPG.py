"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.

torch实现DDPG算法
"""
import torch
import numpy as np
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cuda:0"

# seed = 1
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.set_default_dtype(torch.float)


# Actor Net
# Actor：输入是state，输出的是一个确定性的action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = torch.tensor(action_bound).to(device)

        # layer
        self.layer_1 = nn.Linear(state_dim, 100)
        self.layer_2 = nn.Linear(100, 100)
        self.output = nn.Linear(100, action_dim)

    def forward(self, s):
        a = self.layer_1(s)
        a = torch.tanh(a)
        a = self.layer_2(a)
        a = torch.tanh(a)
        a = self.output(a)
        a = torch.tanh(a)
        # 对action进行放缩，实际上a in [-1,1]
        scaled_a = a * self.action_bound
        return scaled_a


# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         # layer
#         self.layer_1 = nn.Linear(state_dim+action_dim, 100)
#         self.output = nn.Linear(100, 1)
#
#     def forward(self, s, a):
#         x = torch.concat((s, a), 1)
#         x = self.layer_1(x)
#         x = torch.relu(x)
#         q_val = self.output(x)
#         return q_val
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # layer
        self.layer_1 = nn.Linear(state_dim+action_dim, 100)
        self.layer_2 = nn.Linear(100, 100)
        self.output = nn.Linear(100, 1)

    def forward(self, s, a):
        s = torch.cat((s, a), dim=1)
        a = self.layer_1(s)
        a = self.layer_2(torch.tanh(a))
        q_val = self.output(torch.tanh(a))
        return q_val


# Deep Deterministic Policy Gradient
class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, replacement, memory_capacticy=10000, gamma=0.99, lr_a=0.00001,
                 lr_c=0.0001, batch_size=64):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacticy = memory_capacticy
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size
        self.episode = 0
        self.best_loss = 0

        # 记忆库
        self.memory = np.zeros((memory_capacticy, state_dim * 2 + action_dim + 1))
        self.pointer = 0
        # 定义 Actor 网络
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_bound).to(device)
        # 定义 Critic 网络
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()

    def sample(self):
        indices = np.random.choice(self.memory_capacticy, size=self.batch_size)
        return self.memory[indices, :]

    def choose_action(self, s):
        # s = torch.FloatTensor(s).to(device)
        action = self.actor(s)
        return action.item()

    def learn(self):
        for it in range(5):
            # 从记忆库中采样batch data
            bm = self.sample()
            bs = torch.FloatTensor(bm[:, :self.state_dim]).to(device)
            ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim]).to(device)
            br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim]).to(device)
            bs_ = torch.FloatTensor(bm[:, -self.state_dim:]).to(device)

            # 训练critic
            a_ = self.actor_target(bs_)
            q_ = self.critic_target(bs_, a_)
            q_target = br + self.gamma * q_
            q_eval = self.critic(bs, ba)
            td_error = self.mse_loss(q_target, q_eval)
            self.copt.zero_grad()
            td_error.backward()
            self.copt.step()

            # 训练Actor
            a = self.actor(bs)
            q = self.critic(bs, a)
            a_loss = -torch.mean(q)
            self.aopt.zero_grad()
            a_loss.backward(retain_graph=True)
            self.aopt.step()

            # soft replacement and hard replacement
            # 用于更新target网络的参数
            if self.replacement['name'] == 'soft':
                # soft的意思是每次learn的时候更新部分参数
                tau = self.replacement['tau']
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    a = self.actor.state_dict()[al[0] + '.weight']
                    al[1].weight.data.mul_((1 - tau))
                    al[1].weight.data.add_(tau * self.actor.state_dict()[al[0] + '.weight'])
                    al[1].bias.data.mul_((1 - tau))
                    al[1].bias.data.add_(tau * self.actor.state_dict()[al[0] + '.bias'])
                for cl in c_layers:
                    cl[1].weight.data.mul_((1 - tau))
                    cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0] + '.weight'])
                    cl[1].bias.data.mul_((1 - tau))
                    cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0] + '.bias'])

            else:
                # hard的意思是每隔一定的步数才更新全部参数
                if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                    self.t_replace_counter = 0
                    a_layers = self.actor_target.named_children()
                    c_layers = self.critic_target.named_children()
                    for al in a_layers:
                        al[1].weight.data = self.actor.state_dict()[al[0] + '.weight']
                        al[1].bias.data = self.actor.state_dict()[al[0] + '.bias']
                    for cl in c_layers:
                        cl[1].weight.data = self.critic.state_dict()[cl[0] + '.weight']
                        cl[1].bias.data = self.critic.state_dict()[cl[0] + '.bias']

                self.t_replace_counter += 1

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacticy
        self.memory[index, :] = transition
        self.pointer += 1


