import torch
# import torch.nn as nn
import torch.optim as optim
# import numpy as np
# import random
import model
import math
import torch.nn.functional as F
import numpy as np



class ALGO():
    def __init__(self, device, inputsize, outsize = None):
        self.inputsize = inputsize
        self.outsize = outsize
        self.device = device
        self.gamma = 0.7
        self.first_lr = 0.0001
        

        self.Qvalue = model.DQN(
            model.Net(inputsize, outsize), inputsize, gamma = self.gamma, lr = self.first_lr
            )
        self.TargetQ = model.DQN(
            model.Net(inputsize, outsize), inputsize, gamma= self.gamma, lr = self.first_lr
            )

        
        self.optimizer = optim.Adam(self.Qvalue.parameters(), lr = self.first_lr)
        self.TargetOptim = optim.Adam(self.TargetQ.parameters(), lr = self.first_lr)

        self.Qvalue.to(device)
        self.TargetQ.to(device)
        self.max_reward = float('-inf')
        # self.record = []
        try:
            self.Qvalue.load_state_dict(torch.load('model.pth', weights_only=False))
            self.optimizer.load_state_dict(torch.load('optimizer.pth', weights_only=False))
            print("成功加载已有Q模型！")
            self.TargetQ.load_state_dict(torch.load('target.pth', weights_only=False))
            self.TargetOptim.load_state_dict(torch.load('targetoptimizer.pth', weights_only=False))
            print("成功加载已有目标模型！")
        except:
            print("未检测到已有模型！将重新训练")



    def predictQ(self, obs):
        "获取Q(s, a1), Q(s, a2)..."
        # tensor = torch.tensor(obs, dtype=torch.int)
        # obs = torch.FloatTensor(obs).to(self.device) / 3.0
        # 转换为 float32 并归一化
        obs = (
            torch.FloatTensor(obs)
            .unsqueeze(0)  
            .unsqueeze(0)  
            .to(self.device)
            / 3.0
        )
        return self.Qvalue.value(obs)
    
    def predict_taregt(self, obs):
        return self.TargetQ.value(obs)

    def calculate_distance(self, head_x, head_y, food_x, food_y):
        return math.sqrt((head_x - food_x) ** 2 + (head_y - food_y) ** 2)

    def learn(self, obs, save):
        '''
        info = [action, game_run, Length_of_snake, s', [eat, food[], [head_x, head_y], [old_headx, old_heady]], s]
        '''
        states = obs[5]
        actions = obs[0]
        next_states = obs[3]
        dones = not obs[1]
        length = obs[2]
        small = obs[4]


        rewards = 0.5  # 增加存活奖励
        if small[0]:
            rewards += 50  # 提高吃到食物奖励
        old_dis = abs(small[3][0] - small[1][0]) + abs(small[3][1] - small[1][1])  # 改用曼哈顿距离
        new_dis = abs(small[2][0] - small[1][0]) + abs(small[2][1] - small[1][1])
        rewards += 2.0 * (old_dis - new_dis)  # 加大距离差系数
        if dones:
            rewards -= 50  


        # old_dis = self.calculate_distance(small[3][0], small[3][1], small[1][0], small[1][1])
        # new_dis = self.calculate_distance(small[2][0], small[2][1], small[1][0], small[1][1])

        if rewards > self.max_reward:
            self.max_reward = rewards
        # states = torch.FloatTensor(states).view(-1, self.inputsize).to(self.device) / 3.0
        # next_states = torch.FloatTensor(next_states).view(-1, self.inputsize).to(self.device) / 3.0
        states = (
            torch.FloatTensor(states)                # 转换为张量 (shape: [30, 30])
            .unsqueeze(0)                            # 添加批次维度 → [1, 30, 30]
            .unsqueeze(0)                            # 添加通道维度 → [1, 1, 30, 30]
            .to(self.device)                         # 转移至设备（GPU/CPU）
            / 3.0                                    # 归一化
        )

        next_states = (
            torch.FloatTensor(next_states)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
            / 3.0
        )
        

        current_q = self.Qvalue.value(states)  

        actions = torch.tensor([actions], dtype=torch.long).unsqueeze(1).to(self.device)  
        current_q_selected = current_q.gather(1, actions)  

        with torch.no_grad():
            # 使用目标网络计算下一个状态的Q值
            next_q_values = self.TargetQ.value(next_states)
            max_next_q = next_q_values.max(1)[0].unsqueeze(1)  # 取每行最大值
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
    
        loss = F.smooth_l1_loss(current_q_selected, target_q)
        

        
        # 梯度裁剪（防止震荡）
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

        print(f"loss:{loss.item()}  || reward:{rewards} || max_reward:{self.max_reward}")

    def save_model(self):
        torch.save(self.Qvalue.state_dict(), 'model.pth')
        torch.save(self.optimizer.state_dict(), 'optimizer.pth')
        

    def save_target_model(self):
        torch.save(self.TargetQ.state_dict(), 'target.pth')
        torch.save(self.TargetOptim.state_dict(), 'targetoptimizer.pth')

    def show_model(self):
        try:
            self.Qvalue.load_state_dict(torch.load('model.pth'))
            self.optimizer.load_state_dict(torch.load('optimizer.pth'))
            print("成功加载已有模型！")
        except:
            print("未检测到已有模型！")
            quit
    

        
