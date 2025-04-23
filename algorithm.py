import torch
# import torch.nn as nn
import torch.optim as optim
# import numpy as np
# import random
import model
import math
import torch.nn.functional as F
import numpy as np
import copy


class ALGO():
    def __init__(self, device, inputsize, outsize = None):
        self.inputsize = inputsize
        self.outsize = outsize
        self.device = device
        self.gamma = 0.8
        self.first_lr = 0.0001
        #初始状态需要同步参数
        net = model.Net(inputsize, outsize)
        self.Qvalue = model.DQN(
            net
            )
        self.TargetQ = model.DQN(
            copy.deepcopy(net)
            )
        self.optimizer = optim.Adam(self.Qvalue.parameters(), lr = self.first_lr)
        # self.TargetOptim = optim.Adam(self.TargetQ.parameters(), lr = self.first_lr)

        self.Qvalue.to(device)
        self.TargetQ.to(device)
        self.max_reward = float('-inf')
        # self.record = []
        try:
            self.Qvalue.load_state_dict(torch.load('model.pth', weights_only=False))
            self.optimizer.load_state_dict(torch.load('optimizer.pth', weights_only=False))
            print("成功加载已有Q模型及参数！")
            self.TargetQ.load_state_dict(torch.load('target.pth', weights_only=False))
            print("成功加载已有目标模型！")
        except:
            print("未检测到已有模型！将重新训练")



    def predictQ(self, data):
        "获取Q(s, a1), Q(s, a2)..."
        obs = data.copy()
        
        # return self.Qvalue.value(torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device))
        return self.Qvalue.value(torch.FloatTensor(obs).to(self.device))
    
    def predict_taregt(self, data):
        obs = data.copy()
        
        # return self.TargetQ.value(torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device))
        return self.TargetQ.value(torch.FloatTensor(obs).to(self.device))

    def calculate_distance(self, head_x, head_y, food_x, food_y):
        return math.sqrt((head_x - food_x) ** 2 + (head_y - food_y) ** 2)

    def learn(self, data, save, rewards):
        '''
        info = [action, game_run, Length_of_snake, s', [eat, food[], [head_x, head_y], [old_headx, old_heady]], s]
        '''
        obs = data.copy()

        states = obs[5]
        actions = obs[0]
        next_states = obs[3]
        dones = not obs[1]
        length = obs[2]
        small = obs[4]

        self.max_reward = max(rewards, self.max_reward)
        
        # state_tensor = torch.FloatTensor(states).unsqueeze(0).unsqueeze(0).to(self.device)
        # current_q = self.Qvalue.value(state_tensor)[0][actions]

        # # 计算目标Q值
        # next_state_tensor = torch.FloatTensor(next_states).unsqueeze(0).unsqueeze(0).to(self.device)
        # next_q = self.TargetQ.value(next_state_tensor).max(1)[0]

        state_tensor = torch.FloatTensor(states).to(self.device)
        current_q = self.Qvalue.value(state_tensor)[actions]
        next_state_tensor = torch.FloatTensor(next_states).to(self.device)
        next_q = max(self.TargetQ.value(next_state_tensor))

        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q, target_q)
        
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.Qvalue.parameters(), 1.0)
        if save:
            with open('loss.txt', 'a', encoding='utf-8') as f:
                f.write(str(loss.item()) + '\n')
            with open('reward.txt', 'a', encoding='utf-8') as f:
                f.write(str(rewards) + '\n')
            with open('max_reward.txt', 'a', encoding='utf-8') as f:
                f.write(str(self.max_reward) + '\n')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print(f"loss:{loss.item()}  || reward:{rewards} || max_reward:{self.max_reward}")

    def save_model(self):
        torch.save(self.Qvalue.state_dict(), 'model.pth')
        torch.save(self.optimizer.state_dict(), 'optimizer.pth')
        

    def save_target_model(self):
        torch.save(self.TargetQ.state_dict(), 'target.pth')
        # torch.save(self.TargetOptim.state_dict(), 'targetoptimizer.pth')

    def show_model(self):
        try:
            self.Qvalue.load_state_dict(torch.load('model.pth'))
            self.optimizer.load_state_dict(torch.load('optimizer.pth'))
            print("成功加载已有模型！")
        except:
            print("未检测到已有模型！")
            quit
    

        
