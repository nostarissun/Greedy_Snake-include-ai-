import agent
import numpy as np
import copy
import random

class ENV:
    def __init__(self, episodes, how_much_save, how_much_show, epsilon, guide_num):  
        '''
        蛇头为1
        蛇身2
        食物3
        '''
        self.blocks = 15
        self.save_num = how_much_save
        self.show_num = how_much_show
        self.ai = agent.AGENT(self.blocks, epsilon)
        self.max_length = 1
        self.episodes = episodes
        self.page = np.zeros((self.blocks, self.blocks), int)
        # self.act = {
        #     'L': 0,
        #     'R': 1,
        #     'U': 2,
        #     'D': 3
        # }
        self.guide = guide_num
        self.last_dir = 'up'
        self.act = {
            'head_left' : 1,
            'head_right' : 2, 
            'no_change' : 0
        }

    def main(self, show=True):
        length_counter = {}
        for episode in range(self.episodes):
            self.last_dir = 'up'
            print("\033[91m" + "="*80 + "\033[0m")
            print(f"\033[91mEpisode: {episode + 1}\033[0m")
            if not show:
                self.ai.learn(episode)
            # 初始化游戏状态
            run = True
            obs = self.page.copy()
            headx, heady = self.blocks//2, self.blocks//2
            snake = [[headx, heady]]
            foodx, foody = self.make_food(snake)
            obs[foody][foodx] = 3
            obs[heady][headx] = 1
            length = 1
            full = False
            # last_move = 'up' #up
            x_change = 0
            y_change = 0
            rewards = 0
            action = None
            last_obs = None

            acts = {}
            old_headx = headx
            old_heady = heady
            # episode_info = []

            # num = 0
            while run:
                
                if show or (episode % self.show_num == 0 and episode != 0):
                    self.show(obs)
                
                rewards = 0
                # rewards -= 0.5 * num 
 
                if episode < self.guide:
                    old_dis = abs(old_heady - foody) + abs(old_headx - foodx)  
                    new_dis = abs(headx - foodx) + abs(heady - foody)
                    rewards += 0.02 * (old_dis - new_dis) 
 

                last_obs = obs.copy()
                action = self.ai.predict(obs.copy(), False, episode)
                # print(action)
                # 处理方向变化
                dx, dy, valid, self.last_dir = self._handle_movement(action, headx, heady)

                if not valid:
                    run = False
                    break
                if dx == 0 and dy == 0:
                    print("dx,dy wrong")
                    quit
                # 更新头部位置
                new_headx = headx + dx
                new_heady = heady + dy
                
                # 碰撞检测
                if self._check_collision(new_headx, new_heady, snake):
                    run = False
                    break
                
                snake.append([new_headx, new_heady])
                obs[new_heady][new_headx] = 1
                eat = (new_headx == foodx and new_heady == foody)
                if not eat:
                    tail = snake.pop(0)
                    obs[tail[1]][tail[0]] = 0
                else:
                    # rewards += 10
                    rewards += 1 + (length - 1) * 0.2
                    foodx, foody = self.make_food(snake)
                    obs[foody][foodx] = 3
                    length += 1
                    if len(snake) > 1:
                        obs[snake[-2][1]][snake[-2][0]] = 2
                if not show:
                    info = [action, run, length, obs.copy(), 
                       [eat, [foodx, foody], snake[-1], [headx, heady]], last_obs]
                    self.ai.to_pool([info, self.blocks, rewards])
                    # episode_info.append([info, self.blocks, rewards])
                    
                    
                
                old_headx, old_heady = headx, heady
                # 更新头部坐标
                headx, heady = new_headx, new_heady
                
                # 检查是否填满
                if 0 not in obs:
                    full = True
                    break
                if action in acts:
                    acts[action] += 1
                else:
                    acts[action] = 1 
            
            if length in length_counter:
                length_counter[length] += 1
            else:
                length_counter[length] = 1
   
            print(f"\033[91mlength: {length}  maxLength:{self.max_length} action:{acts}\033[0m")
            with open('length_num.txt', 'a', encoding='utf-8') as f:
                f.write(str(length_counter) + '\n')
            with open('length.txt', 'a', encoding='utf-8') as f:
                f.write(str(length) + '\n')
            with open('maxlength.txt', 'a', encoding='utf-8') as f:
                f.write(str(self.max_length) + '\n')

            self.max_length = max(length, self.max_length)

            # 定期保存模型
            if episode == self.episodes - 1 or (episode > 0 and episode % self.save_num) == 0:
                self.ai.algo.save_model()
                self.ai.algo.save_target_model()

            if run == False and not show:
                rewards -= 1 + 0.2 * (length // 2)
                info = [action, False, length, obs.copy(), 
                    [False, [foodx, foody], snake[-1], [headx, heady]], last_obs]
                self.ai.to_pool([info, self.blocks, rewards])

        print(length_counter)
    def _handle_movement(self, action, current_x, current_y):
        """处理移动方向和有效性验证"""
        last_key = self.last_dir
        now_dir = last_key
        dx, dy = 0, 0
        move = {
            'left' : (-1, 0),
            'right' : (1, 0),
            'up' : (0, -1),
            'down' : (0, 1)
        }
        if action == self.act['no_change']:
            if last_key == 'up':
                dx, dy = move['up']
            elif last_key == 'down':
                dx, dy = move['down']
            elif last_key == 'left':
                dx, dy = move['left']
            else:
                dx, dy = move['right']
        elif action == self.act['head_left']:
            if last_key == 'up' or last_key == 'down':
                dx, dy = move['left']
                now_dir = 'left'
            elif last_key == 'left':
                dx, dy = move['down']
                now_dir = 'down'
            else:
                dx, dy = move['up']
                now_dir = 'up'
        else:
            if last_key == 'up' or last_key == 'down':
                dx, dy = move['right']
                now_dir = 'right'
            elif last_key == 'left':
                dx, dy = move['up']
                now_dir = 'up'
            else:
                dx, dy = move['down']
                now_dir = 'down'

        # print(last_key, now_dir)
        # 边界检查
        new_x = current_x + dx
        new_y = current_y + dy
        if not (0 <= new_x < self.blocks and 0 <= new_y < self.blocks):
            return 0, 0, False, last_key
        
        return dx, dy, True, now_dir

    def _check_collision(self, x, y, snake):
        """碰撞检测优化"""
        if not (0 <= x < self.blocks and 0 <= y < self.blocks):
            return True
        return any(segment[0] == x and segment[1] == y for segment in snake)

    def make_food(self, snake):
        """改进的食物生成算法"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(0, self.blocks-1)
            y = random.randint(0, self.blocks-1)
            if [x, y] not in snake:
                return x, y
        # 极端情况处理：返回第一个可用的位置
        for x in range(self.blocks):
            for y in range(self.blocks):
                if [x, y] not in snake:
                    return x, y
        # return -1, -1  # 游戏结束

    def show(self, page):
        """增强型显示支持多颜色"""
        color_map = {
            1: "\033[91m",  # 红色蛇头
            2: "\033[94m",  # 蓝色蛇身
            3: "\033[92m"   # 绿色食物
        }
        for row in page:
            for val in row:
                if val in color_map:
                    print(f"{color_map[val]}{val}\033[0m", end=' ')
                else:
                    print(0, end=' ')
            print()
        print(f"\033[93m{'='*(self.blocks*2)}\033[0m")  # 黄色分隔线