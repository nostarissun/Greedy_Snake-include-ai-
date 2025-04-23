import queue
import algorithm
import random
import torch
import numpy as np
import copy
import math

class AGENT():
    def __init__(self, inputsize, epsilon):
        self.output_size = 3
        if torch.cuda.is_available():
            self.algo = algorithm.ALGO('cuda', inputsize, self.output_size)
            print("use gpu!")
        else:
            print("使用CPU进行训练！")
            self.algo = algorithm.ALGO('cpu', inputsize, self.output_size)
        
        self.episodes = 1
        self.max_size = 400
        self.batch_size = 20
        self.buffer = queue.deque(maxlen = self.max_size)

        self.epsilon_start = epsilon
 

        
    def to_pool(self, info):
        '''
        info = [action, game_run, Length_of_snake, s', [eat, [foodx, foody], [head_x, head_y], [old_headx, old_heady]], s]
        act = {'L': 0,
            'R': 1,
            'U': 2,
            'D': 3}
        '''
        self.buffer.append(info)
        # action = info[0]
        # game_run = info[1]
        # length = info[2]
        # next_state = info[3]
        # eat_info = info[4]
        # state = info[5]
        # # 定义旋转后的动作映射
        # action_maps = [
        #     {0:0, 1:1, 2:2, 3:3},  # k=0 (无旋转)
        #     {0:2, 1:3, 2:1, 3:0},  # k=1 (顺时针90°)
        #     {0:1, 1:0, 2:3, 3:2},  # k=2 (顺时针180°)
        #     {0:3, 1:2, 2:0, 3:1},  # k=3 (顺时针270°)
        # ]

        # for k in range(0, 4):
        #     if len(self.buffer) == self.max_size:
        #         self.buffer.pop()
        #     rotated_state = np.rot90(state, k).copy()
        #     rotated_next_state = np.rot90(next_state, k).copy()
        #     new_action = action_maps[k][action]
        #     new_eat_info = copy.deepcopy(eat_info)

        #     # 处理食物坐标
        #     fx, fy = new_eat_info[1][0], new_eat_info[1][1]
        #     if k == 1:
        #         # 顺时针90
        #         new_eat_info[1][0], new_eat_info[1][1] = blocks - fy, fx
                
        #     elif k == 2:
        #         # 顺时针180度
                
        #         new_eat_info[1][0], new_eat_info[1][1] = blocks - fy, blocks - fx
        #     elif k == 3:
        #         # 顺时针270度
        #         new_eat_info[1][0], new_eat_info[1][1] = fy, blocks - fx
                
            
        #     # 处理蛇头和旧蛇头坐标（同食物逻辑）
        #     # head坐标处理
        #     hx, hy = new_eat_info[2][0], new_eat_info[2][1]
        #     if k == 1:
        #         new_eat_info[2][0], new_eat_info[2][1] = hy, blocks - hx
        #     elif k == 2:
        #         new_eat_info[2][0], new_eat_info[2][1] = blocks - hx, blocks - hy
        #     elif k == 3:
        #         new_eat_info[2][0], new_eat_info[2][1] = blocks - hy, hx
            
        #     # old_head处理同理
        #     ohx, ohy = new_eat_info[3][0], new_eat_info[3][1]
        #     if k == 1:
        #         new_eat_info[3][0], new_eat_info[3][1] = ohy, blocks - ohx
        #     elif k == 2:
        #         new_eat_info[3][0], new_eat_info[3][1] = blocks - ohx, blocks - ohy
        #     elif k == 3:
        #         new_eat_info[3][0], new_eat_info[3][1] = blocks - ohy, ohx
            

       
        #     self.buffer.append([(new_action, game_run, length, rotated_next_state, new_eat_info, rotated_state), reward])

    def predict(self, obs, show, cnt):
        value = self.algo.predictQ(obs)
        # print(value)
        if show:
            v = value[0]
            # print(max_value)

            max_indices = []
            # print(v)
            max_v = max(v)
            for i in range(self.output_size):
                if v[i] == max_v:
                    max_indices.append(i)

            res = random.choice(max_indices)
            return res
        else:
            # print(f"epsilon:{self.epsilon_start}")
            rand = np.random.uniform(0, 1)
            if rand <= self.epsilon_start:
                # print("！随机选择！")
                return random.randint(0, 2)
            else:
                v = value[0]
                # print(max_value)

                max_indices = []
                # print(v)
                max_v = max(v)
                for i in range(self.output_size):
                    if v[i] == max_v:
                        max_indices.append(i)

                res = random.choice(max_indices)
                return res
                    
        
    def learn(self, cnt):
        '''
        同步一下，target和model的参数
        '''
        if (cnt + 1) % 50 == 0:
            self.algo.TargetQ.model.load_state_dict(self.algo.Qvalue.model.state_dict())

            
        if len(self.buffer) <= self.batch_size * 2:
            # print("样本不足，不进行训练！")
            return
        else:
            batch = random.sample(self.buffer, self.batch_size)
        i = 1
        for datas in batch:
            # print(f"正在训练第{i}组数据")
            if i == self.batch_size:
                self.algo.learn(datas[0], True, datas[2])
            else:
                self.algo.learn(datas[0], False, datas[2])

            i += 1






