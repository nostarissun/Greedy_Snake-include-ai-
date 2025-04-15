import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
'''
将当前状态作为输入张量，传递给神经网络，神经网络输出q值，以供算法做出决策'
'''
# 定义模型
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQN():
    def __init__(self, model, act_dim = None, gamma = None, lr = None):
        '''
        act_dim(int): action空间维度
        gamma: reward的衰减因子
        '''
        self.model = model
        # if target_model == None:
        self.target_model = copy.deepcopy(self.model)

        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
    
    
    
    def value(self):
        return self.model.forward(self.obs)
    
    def sync_target(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)  # 直接复制参数



    