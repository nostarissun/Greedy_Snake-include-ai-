import torch
# import torch.nn as nn
import torch.optim as optim
# import numpy as np
# import random
import model



class ALGO():
    def __init__(self, device, inputsize, outsize = None):
        self.outsize = outsize
        self.device = device
   
        self.first_lr = 0.01
        
        self.neural_net = model.Net(inputsize, outsize)
        self.Qvalue = model.DQN(inputsize, gamma = 0.88, lr = self.first_lr)
        self.TargetQ = model.DQN(inputsize, gamma= 0.88, lr = self.first_lr)

        
        self.optimizer = optim.Adam(self.Qvalue.parameters(), lr = self.first_lr)
        self.TargetOptim = optim.Adam(self.TargetQ.parameters(), lr = self.first_lr)

        self.Qvalue.to(device)
        self.TargetQ.to(device)
        try:
            self.Qvalue.load_state_dict(torch.load('model.pth'))
            self.optimizer.load_state_dict(torch.load('optimizer.pth'))
            print("成功加载已有Q模型！")
            self.TargetQ.load_state_dict(torch.load('model.pth'))
            self.TargetOptim.load_state_dict(torch.load('optimizer.pth'))
            print("成功加载已有目标模型！")
        except:
            print("未检测到已有模型！将重新训练")
   
    def predictQ(self, obs):
        "获取Q(s, a1), Q(s, a2)..."
        return self.Qvalue.value(obs)
    
    def predict_taregt(self, obs):
        return self.TargetQ.value(obs)

    def learn(self, obs):
        last_act = obs[0]
        game_run = obs[1]
        reward = obs[2]  #蛇身长度
        obs = obs[3]
        if game_run == False:
            reward -= 100
        
        '''
        获取target值
        '''
        next_pred_value = self.TargetQ.value(obs)
        q_target = max(next_pred_value)
        '''
        获取model预测值
        '''
        cur_q_value = self.Qvalue.value(obs)
        '''
        更新参数
        '''
        loss = ((reward + self.gamma * max(cur_q_value)) - q_target) ** 2
        self.Qvalue.optimizer.zero_grad()
        loss.backward()
        self.TargetQ.optimizer.step()

    def save_model(self, model):
        torch.save(model.state_dict(), 'model.pth')
        optim.load_state_dict(torch.load('optimizer.pth'))

    def save_target_model(self, target_model):
        torch.save(target_model.state_dict(), 'model.pth')
        optim.load_state_dict(torch.load('optimizer.pth'))

    def show_model(self):
        try:
            self.Qvalue.load_state_dict(torch.load('model.pth'))
            optim.load_state_dict(torch.load('optimizer.pth'))
            print("成功加载已有模型！")
        except:
            print("未检测到已有模型！")
            quit
    

        
