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
# class Net(nn.Module):
    # def __init__(self, grid_size, output_size):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    #     self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    #     self.fc1 = nn.Linear(64 * 7 * 7, 256)
    #     self.fc2 = nn.Linear(256, output_size)

    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     # x = F.max_pool2d(x, 2)
    #     x = F.relu(self.conv2(x))
    #     # x = F.max_pool2d(x, 2)
    #     x = F.relu(self.conv3(x))
    #     x = x.view(x.size(0), -1)
    #     x = F.relu(self.fc1(x))
    #     return self.fc2(x)
    # def __init__(self, input_dim, hidden_dim, output_dim):
    #     super().__init__()
    #     self.fc1 = nn.Linear(input_dim, hidden_dim)
    #     self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    #     self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    #     self.fc4 = nn.Linear(hidden_dim, output_dim)
    #     self.relu = nn.ReLU()
        
    # def forward(self, x):
    #     l1 = self.relu(self.fc1(x.float()))
    #     l2 = self.relu(self.fc2(l1))
    #     l3 = self.relu(self.fc3(l2))
    #     l4 = self.fc4(l3)
    #     return l4

class Net(nn.Module):
    def __init__(self, grid_size=15, output_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        # 计算展平维度：
        self.fc1 = nn.Linear(8 * 7 * 7, 16)
        self.fc2 = nn.Linear(16, output_size)
        
    def forward(self, x):
        # print(x)
        x = F.relu(self.conv1(x))
        # print(x)
        x = self.pool1(x)  # 输出尺寸：7x7
        # print(x)
        x = F.relu(self.conv2(x))  # 输出尺寸：7x7
        # print(x)
        x = x.view(x.size(0), -1)  # 展平为 8x7x7=392
        # print(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    
    def value(self, data):
        obs = data
        return self.model.forward(obs)
        # input_tensor = torch.from_numpy(obs).float()  
        # input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  
        # print(input_tensor)
        # output = self.model.forward(input_tensor).detach().cpu()
        # return tuple(output.squeeze(0).numpy().tolist())
    




    