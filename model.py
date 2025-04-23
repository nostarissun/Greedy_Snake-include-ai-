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
#     def __init__(self, grid_size=15, output_size=3):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
#         self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
#         # 计算展平维度：
#         self.fc1 = nn.Linear(8 * 7 * 7, 16)
#         self.fc2 = nn.Linear(16, output_size)
        
#     def forward(self, x):
#         # print(x)
#         x = F.relu(self.conv1(x))
#         # print(x)
#         x = self.pool1(x)  # 输出尺寸：7x7
#         # print(x)
#         x = F.relu(self.conv2(x))  # 输出尺寸：7x7
#         # print(x)
#         x = x.view(x.size(0), -1)  # 展平为 8x7x7=392
#         # print(x)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
class Net(nn.Module):
    def __init__(self, inputsize, outputsize):
        super().__init__()
        self.fc1 = nn.Linear(inputsize * inputsize, inputsize * 8)
        self.fc2 = nn.Linear(inputsize * 8, inputsize * 2)
        self.fc3 = nn.Linear(inputsize * 2, inputsize)
        self.fc4 = nn.Linear(inputsize, outputsize)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        l1 = self.relu(self.fc1(x))
        l2 = self.relu(self.fc2(l1))
        l3 = self.relu(self.fc3(l2))
        l4 = self.fc4(l3)
        return l4

class DQN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    
    def value(self, data):
        obs = copy.deepcopy(data).flatten()
        return self.model.forward(obs)
        # input_tensor = torch.from_numpy(obs).float()  
        # input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  
        # print(input_tensor)
        # output = self.model.forward(input_tensor).detach().cpu()
        # return tuple(output.squeeze(0).numpy().tolist())
    




    