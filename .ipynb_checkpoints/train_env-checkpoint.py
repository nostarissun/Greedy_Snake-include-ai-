import agent
import numpy as np
import copy
import random

class ENV:
    def __init__(self, episodes, how_mush_save, how_mush_show):
        '''
        蛇头为1
        蛇身2
        食物3
        '''
        self.blocks = 30
        self.save_num = how_mush_save
        self.show_num = how_mush_show
        self.ai = agent.AGENT(self.blocks * self.blocks)
        
        self.episodes = episodes
        self.episode = 1
        self.page = np.zeros((self.blocks, self.blocks), float)
        self.act = {
            'L': 0,
            'R': 1,
            'U': 2,
            'D': 3
                }
    def main(self, show = True):
        'episodes局游戏'
        for i in range(self.episodes):
            '每一局游戏'
            
            print("```````````````````````````````````````````````````````")
            print(f"episode:{self.episode}")
            
            run = True
            obs = self.page.copy()
            last_key = None
            headx = self.blocks // 2
            heady = self.blocks // 2
            snake = [(headx, heady)]
            foodx, foody = self.make_food(snake)
            obs[foody][foodx] = 3
            obs[heady][headx] = 1
            
            length = 1
            full = False
            while run:
                '坐标全部取索引'
                if show:
                    self.show()
                else:
                    if i % self.show_num:
                        self.show()
                last_obs = obs.copy()
                action = self.ai.predict(obs, False, i)
                if action == self.act["L"]:
                    if last_key != None and last_key == self.act["R"]:
                        pass
                    else:
                        x_change = -1
                        y_change = 0
                        last_key = self.act["L"]
                elif action == self.act["R"]:
                    if last_key != None and last_key == self.act["L"]:
                        pass
                    else:
                        x_change = 1
                        y_change = 0
                        last_key = self.act["R"]
                elif action == self.act["U"]:
                    if last_key != None and last_key == self.act["D"]:
                        pass
                    else:
                        y_change = -1
                        x_change = 0
                        last_key = self.act["U"]
                elif action == self.act["D"]:
                    if last_key != None and last_key == self.act["U"]:
                        pass
                    else:
                        y_change = 1
                        x_change = 0
                        last_key = self.act["D"]
                else:
                    pass
                
                new_headx = headx + x_change
                new_heady = heady + y_change
                if new_headx >= self.blocks or new_headx < 0 or new_heady >= self.blocks or new_heady < 0:
                    run = False
                for i in range(length):
                    if new_headx == snake[i][0] and new_heady == snake[i][1]:
                        run = False
                if run == False:
                    break
                snake.append((new_headx, new_heady))
                eat = True
                if new_headx != foodx or new_heady != foody:
                    del snake[0]
                    eat = False
                obs[new_heady][new_headx] = 1
                obs[snake[-2][1]][snake[-2][0]] = 2
                if eat == True:
                    foodx, foody = self.make_food(snake)
                    obs[foody][foodx] = 3
                if 0 not in obs:
                    full = True
                
                if not show:
                    info = [action, run, length, obs, [eat, [foodx, foody], snake[-1], snake[-2]], last_obs]
                    self.ai.to_pool(info)

                if full:
                    break

            if i % self.save_num:
                self.ai.algo.save_model()
                self.ai.algo.save_target_model()
            
            
                    
    def make_food(self, snake):
        '''
        除掉蛇身1-blocks - 1随机选择 x , y
        '''
        x, y = random.randrange(1, self.blocks - 1), random.randrange(1, self.blocks -1)
        while (x, y) in snake: 
            x, y = random.randrange(1, self.blocks - 1), random.randrange(1, self.blocks -1)    
        return x, y

    def show(self, page):
        '''
        输出目标数组page
        '''
        print(page)

