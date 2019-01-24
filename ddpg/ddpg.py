"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
from tensorboardX import SummaryWriter

#####################  hyper parameters  ####################

MAX_EPISODES = 300
MAX_EP_STEPS = 200
LR_A = 0.0001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################

class Actor_Net(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super(Actor_Net, self).__init__()
        self.a_bound = a_bound
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.001)   # initialization
        self.relu = torch.nn.ReLU()
        # self.fc1_bn = nn.BatchNorm1d(50)
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.001)   # initialization
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        if(isinstance(x,np.ndarray)):
            x = torch.tensor(x).type(torch.FloatTensor)
        if(isinstance(self.a_bound, np.ndarray)):
            self.a_bound = torch.tensor(self.a_bound).type(torch.FloatTensor)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.tanh(x)

        actions_value = x.mul(self.a_bound)

        return actions_value

class Critic_Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic_Net, self).__init__()
        self.w1_s = torch.nn.Parameter(torch.zeros(s_dim, 30))
        self.w1_a = torch.nn.Parameter(torch.zeros(a_dim, 30))
        self.b = torch.nn.Parameter(torch.zeros(1, 30))
        self.fc = nn.Linear(30,1)
        self.fc.weight.data.normal_(0, 0.001)

    def forward(self, s, a):
        if(isinstance(s,np.ndarray)):
            s = torch.tensor(s).type(torch.FloatTensor)
        if (isinstance(a, np.ndarray)):
            a = torch.tensor(a).type(torch.FloatTensor)

        out = s.mm(self.w1_s) + a.mm(self.w1_a) + self.b
        out = self.fc(out)
        return out

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):

        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.memory_pointer = 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        self.actor_eval = Actor_Net(self.s_dim, self.a_dim, self.a_bound)
        self.actor_target = Actor_Net(self.s_dim, self.a_dim, self.a_bound)
        self.actor_eval_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_A)

        self.critic_eval = Critic_Net(self.s_dim, self.a_dim)
        self.critic_target = Critic_Net(self.s_dim, self.a_dim)
        self.critic_eval_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_C)

    def soft_replace(self):#缓慢替换target网络
        for eval_param, target_param in zip(self.actor_eval.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + eval_param.data * TAU)
        for eval_param, target_param in zip(self.critic_eval.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + eval_param.data * TAU)

    def eval(self):
        self.actor_eval.eval()
        self.actor_target.eval()
        self.critic_eval.eval()
        self.critic_target.eval()

    def choose_action(self, s):
        # self.eval()
        action = self.actor_eval.forward(s.reshape(1, -1))
        if(not isinstance(action, np.ndarray)):
            action = action.detach().numpy()
        action = action.reshape(-1)
        return action

    def learn(self):
        # soft target replacement
        self.soft_replace()

        sample_index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.s_dim])
        b_a = torch.FloatTensor(b_memory[:, self.s_dim:self.s_dim + self.a_dim])
        b_r = torch.FloatTensor(b_memory[:, -self.s_dim - 1: -self.s_dim])
        b_s_ = torch.FloatTensor(b_memory[:, -self.s_dim:])

        a = self.actor_eval.forward(b_s)      #进行actor_eval的更新
        q = self.critic_eval.forward(b_s, a)
        a_loss = torch.mean(-q)

        self.actor_eval_optim.zero_grad()
        # a_loss.backward(retain_graph=True)
        a_loss.backward()
        self.actor_eval_optim.step()

        q = self.critic_eval.forward(b_s, b_a)   #进行critic_eval的更新
        a_ = self.actor_target.forward(b_s_).detach()
        q_ = self.critic_target.forward(b_s_, a_).detach()
        q_target = b_r + GAMMA * q_
        loss_func = nn.MSELoss()
        td_error = loss_func(q_target, q)

        self.critic_eval_optim.zero_grad()
        td_error.backward()
        self.critic_eval_optim.step()
        return a_loss, td_error

    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, a, [r], s_))
        index = self.memory_pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.memory_pointer += 1


###############################  training  ####################################
# env = gym.make(ENV_NAME)
# env = env.unwrapped
# env.seed(1)
#
# s_dim = env.observation_space.shape[0]
# a_dim = env.action_space.shape[0]
# a_bound = env.action_space.high
#
#
# dummy_input = torch.rand(3,3)
#
# print(dummy_input)
# model = Actor_Net(s_dim, a_dim, a_bound)
# # with SummaryWriter(comment='Actor_Net') as w:
# #     w.add_graph(model, (dummy_input))
#
# C_model = Critic_Net(s_dim, a_dim)
# a_input = model.forward(dummy_input)
# print(a_input)
# print(C_model.forward(dummy_input, a_input))
#
# with SummaryWriter(comment='Critic_Net') as w:
#     w.add_graph(model, (dummy_input))
#     w.add_graph(C_model, (dummy_input, a_input,))


env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
writer = SummaryWriter(log_dir='scalar')
x = 0
for i in range(MAX_EPISODES):
    s = env.reset()

    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.memory_pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            loss, td_error = ddpg.learn()
            writer.add_scalars('scalar/scalars_test', {'a_loss': loss, 'td_error': td_error}, x)
            x += 1
        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)
writer.close()
