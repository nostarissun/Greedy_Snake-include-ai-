import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import torch.nn.functional as F
'''
将当前状态作为输入张量，传递给神经网络，神经网络输出q值，以供算法做出决策'
'''
# 定义模型




# class Net(nn.Module):
#     def __init__(self, grid_size, output_size):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        
    
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)) 
#         self.fc1 = nn.Linear(16 * 7 * 7, 64)
#         self.fc2 = nn.Linear(64, output_size)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)  
#         x = F.relu(self.conv2(x))
#         x = self.adaptive_pool(x)  
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
class Net(nn.Module):
    def __init__(self, grid_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN(nn.Module):
    def __init__(self, model, act_dim = None, gamma = None, lr = None):
        '''
        act_dim(int): action空间维度
        gamma: reward的衰减因子
        '''
        super().__init__()
        self.model = model
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
    
    def value(self, obs):
        return self.model.forward(obs)
    




    