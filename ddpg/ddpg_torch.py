import time
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn

MAX_EPISODES = 300
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
N_HIDDEN_UNIT = 30
RENDER = False
# -----------   declear environment   ----------------------

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
# -----------   infomation of env   --------------------
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_max = env.action_space.high
a_min = env.action_space.low

# ---------   declear ddpg   -----------
class DDPG(object):
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        # actor net
        self.actor_eval = self._build_actor()
        self.actor_target = self._build_actor()
        self.actor_eval_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_A)
        # eval net
        self.critic_eval = self._build_critic()
        self.critic_target = self._build_critic()
        self.critic_eval_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_C)
        # copy target net parameter to eval net
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())

    def _build_critic(self):
        net = nn.Sequential(
            nn.Linear(s_dim + a_dim, N_HIDDEN_UNIT),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
        return net

    def _build_actor(self):
        net = nn.Sequential(
            nn.Linear(s_dim, N_HIDDEN_UNIT),
            nn.ReLU(),
            nn.Linear(N_HIDDEN_UNIT, a_dim),
            nn.Tanh()
        )
        return net

    def choose_action(self, s):
        s = (torch.FloatTensor(s))
        action = self.actor_eval(s).data.numpy() * a_max
        return np.clip(action, a_min, a_max)

    def learn(self):
        sample_index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        b_memory = np.array([self.memory[x] for x in sample_index])
        b_s = torch.FloatTensor(b_memory[:, :s_dim])
        b_a = torch.FloatTensor(b_memory[:, s_dim:s_dim + a_dim])
        b_r = torch.FloatTensor(b_memory[:, -s_dim - 1: -s_dim])
        b_s_ = torch.FloatTensor(b_memory[:, -s_dim:])

        self.actor_learn(b_s)
        self.critic_learn(b_s, b_a, b_r, b_s_)
        self.soft_replace()

    def actor_learn(self,b_s):
        a = self.actor_eval.forward(b_s)  # 进行actor_eval的更新
        ce_s = torch.cat([b_s, a], 1)
        q = self.critic_eval.forward(ce_s)
        a_loss = torch.mean(-q)

        self.actor_eval_optim.zero_grad()
        a_loss.backward()
        self.actor_eval_optim.step()

    def critic_learn(self,b_s,b_a,b_r,b_s_):
        ce_s = torch.cat([b_s,b_a], 1)
        q = self.critic_eval.forward(ce_s)  # 进行critic_eval的更新
        a_ = self.actor_target.forward(b_s_).detach()
        ct_s = torch.cat([b_s_,a_],1)
        q_ = self.critic_target.forward(ct_s).detach()
        q_target = b_r + GAMMA * q_
        loss_func = nn.MSELoss()
        td_error = loss_func(q_target, q)

        self.critic_eval_optim.zero_grad()
        td_error.backward()
        self.critic_eval_optim.step()

    def soft_replace(self):  # 缓慢替换target网络
        for eval_param, target_param in zip(self.actor_eval.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + eval_param.data * TAU)
        for eval_param, target_param in zip(self.critic_eval.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + eval_param.data * TAU)

    def store_transition(self, s, a, r, s_):  # 记忆库存储函数
        record = np.hstack((s, a, r, s_))
        self.memory.append(record)


def main():
    ddpg = DDPG()
    RENDER = False
    var = 3  # control exploration
    for i in range(MAX_EPISODES):
        s = env.reset()  # 重置状态
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            a = ddpg.choose_action(s)  # 选择动作
            a = np.clip(np.random.normal(a, var), -2, 2)    # 添加噪音
            s_, r, done, info = env.step(a)  # 仿真
            ddpg.store_transition(s, a, r, s_)  # 存储记忆库

            if len(ddpg.memory) > MEMORY_CAPACITY - 1:  #学习并减少噪音
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_  # 更新状态
            ep_reward += r

            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -300 : RENDER = True
                break


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print(time_end - time_start, 's')
