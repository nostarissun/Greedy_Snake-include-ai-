import pygame
import time
import random
import agent
import numpy as np
import threading
import copy
'''
游戏框架
'''
class Game:
    pygame.init()

    def __init__(self, train, show):
        
        # 定义颜色
        self.white = (255, 255, 255)
        self.yellow = (255, 255, 102)
        self.black = (0, 0, 0)
        self.red = (213, 50, 80)
        self.green = (0, 255, 0)
        self.blue = (50, 153, 213)
        self.light_gray = (200, 200, 200)
        # 定义游戏窗口尺寸
        self.blocks = 30
        self.block = 20
        self.win_width = self.blocks * self.block + 1 + self.blocks
        self.win_height = self.blocks * self.block + 1 + self.blocks
        self.line_width = 1
        # 创建游戏窗口
        # self.win = pygame.display.set_mode((self.win_width, self.win_height))
        # pygame.display.set_caption('ai贪吃蛇')

        # self.clock = pygame.time.Clock()

        # self.font_style = pygame.font.SysFont("bahnschrift", 25)
        # self.score_font = pygame.font.SysFont("comicsansms", 35)
        self.train = train
        self.show = show
        
        self.ai = agent.AGENT((self.blocks) * (self.blocks))
        
        self.episodes = 6000
        self.episode = 1


        # self.info = []


    def print_score(self, score):
        value = self.score_font.render("your score " + str(score), True, self.red)
        self.win.blit(value, [0, 0])


    def show_snake(self, snake_block, snake_list, Length_of_snake):
        for x in snake_list:
            if Length_of_snake > 1:
                pygame.draw.rect(self.win, self.blue, [x[0] * (self.block + self.line_width) + self.line_width, x[1] * (self.block + self.line_width) + self.line_width, snake_block, snake_block])
            else:
                pygame.draw.rect(self.win, self.black, [snake_list[-1][0] * (self.block + self.line_width) + self.line_width, snake_list[-1][1] * (self.block + self.line_width) + self.line_width, snake_block, snake_block])
            Length_of_snake -= 1

    def message(self,info, color):
        info = self.font_style.render(info, True, color)
        self.win.blit(info, [self.win_width / 6, self.win_height / 3])


    def main_page(self):
        self.win.fill(color=self.white)
        for i in range(self.blocks):
            pygame.draw.line(self.win, self.light_gray, (i * self.block + i, 0), (i * self.block + i, self.win_height), width=self.line_width)
        for i in range(self.blocks):
            pygame.draw.line(self.win, self.light_gray, (0, i * self.block + i), (self.win_width, i * self.block + i), width=self.line_width)


    def make_food(self):
        random_foodx = random.randrange(1, self.blocks)
        foodx = random_foodx * (self.block + self.line_width) + self.line_width
        random_foody = random.randrange(1, self.blocks)
        foody = random_foody * (self.block + self.line_width) + self.line_width
        return foodx, foody


    def run_game(self):
        game_run = True
        train = self.train
        show = self.show
        # snake_move_interval = 80  # 蛇移动的时间间隔（毫秒）
        # last_move_time = pygame.time.get_ticks()
        '''
        蛇头为1
        蛇身2
        食物3
        '''
        head_x = self.blocks // 2
        head_y = self.blocks // 2

        x_change = 0
        y_change = 0
        last_key = None
        snake_List = []
        snake_Head = [head_x, head_y]
        Length_of_snake = 1
        foodx, foody = self.make_food()

        input_info = np.zeros((self.blocks, self.blocks), dtype=int)
        
        input_info[head_y][head_x] = 1
        state = None

        while game_run and self.episode <= self.episodes:
            if train == False:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_run = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            if last_key != None and last_key == pygame.K_RIGHT:
                                pass
                            else:
                                x_change = -1
                                y_change = 0
                                last_key = event.key
                        elif event.key == pygame.K_RIGHT:
                            if last_key != None and last_key == pygame.K_LEFT:
                                pass
                            else:
                                x_change = 1
                                y_change = 0
                                last_key = event.key
                        elif event.key == pygame.K_UP:
                            if last_key != None and last_key == pygame.K_DOWN:
                                pass
                            else:
                                y_change = -1
                                x_change = 0
                                last_key = event.key
                        elif event.key == pygame.K_DOWN:
                            if last_key != None and last_key == pygame.K_UP:
                                pass
                            else:
                                y_change = 1
                                x_change = 0
                                last_key = event.key
                        elif event.key == pygame.K_q:
                            game_run = False
                        else:
                            pass
            else:
                try:
                    print("```````````````````````````````````````````````````````")
                    print(f"episode:{self.episode}")
                    y = (foody - self.line_width) // (self.block + self.line_width)
                    x = (foodx - self.line_width) // (self.block + self.line_width)
                    input_info[y][x] = 3
                    
                    state = copy.deepcopy(input_info)
                    action = self.ai.predict(input_info, self.show, self.episode)
                    print(f'action:{action} || food({x}, {y}) || head({snake_Head[0]}, {snake_Head[1]})')
                    act = {'L': 0,
                        'R': 1,
                        'U': 2,
                        'D': 3}

                    if action == act["L"]:
                        if last_key != None and last_key == act["R"]:
                            pass
                        else:
                            x_change = -1
                            y_change = 0
                            last_key = act["L"]
                    elif action == act["R"]:
                        if last_key != None and last_key == act["L"]:
                            pass
                        else:
                            x_change = 1
                            y_change = 0
                            last_key = act["R"]
                    elif action == act["U"]:
                        if last_key != None and last_key == act["D"]:
                            pass
                        else:
                            y_change = -1
                            x_change = 0
                            last_key = act["U"]
                    elif action == act["D"]:
                        if last_key != None and last_key == act["U"]:
                            pass
                        else:
                            y_change = 1
                            x_change = 0
                            last_key = act["D"]
                    else:
                        pass
                except KeyboardInterrupt:
                    if show == False:
                        self.ai.algo.save_model()
                        self.ai.algo.save_target_model()

            # current_time = pygame.time.get_ticks()
            # if current_time - last_move_time >= snake_move_interval:
                # 检测蛇是否超出边界
            new_head_x = head_x + x_change
            new_head_y = head_y + y_change
            new_head_x_pixel = new_head_x * (self.block + self.line_width) + self.line_width
            new_head_y_pixel = new_head_y * (self.block + self.line_width) + self.line_width
            if new_head_x_pixel >= self.win_width or new_head_x_pixel < 0 or new_head_y_pixel >= self.win_height or new_head_y_pixel < 0:
                game_run = False

            
            snake_Head = [new_head_x, new_head_y]
            snake_List.append(snake_Head)

            # 检测蛇是否碰到自己的身体 
            for i in range(Length_of_snake - 1):
                if snake_Head[0] == snake_List[i][0] and snake_Head[1] == snake_List[i][1]:
                    game_run = False

            if game_run and (train or show):
                input_info[new_head_y][new_head_x] = 1
                input_info[head_y][head_x] = 2



           
            if len(snake_List) > Length_of_snake:
                if show or train:
                    input_info[snake_List[0][1]][snake_List[0][0]] = 0

                del snake_List[0]

            old_headx, old_heady = head_x, head_y
            head_x = new_head_x
            head_y = new_head_y
            # last_move_time = current_time
            #缩进截至处

          
            # self.main_page()
            # pygame.draw.rect(self.win, self.green, [foodx, foody, self.block, self.block])

            # self.show_snake(self.block, snake_List, Length_of_snake)
            # self.print_score(Length_of_snake - 1)

            # pygame.display.update()

            eat = False
            food = [(foodx - self.line_width) // (self.block + self.line_width), (foody - self.line_width) // (self.block + self.line_width)]
            
            head_x_pixel = head_x * (self.block + self.line_width) + self.line_width
            head_y_pixel = head_y * (self.block + self.line_width) + self.line_width
            if head_x_pixel == foodx and head_y_pixel == foody:
                if train or show:
                    input_info[(foody - self.line_width) // (self.block + self.line_width)][(foodx - self.line_width) // (self.block + self.line_width)] = 3
                foodx, foody = self.make_food()
                eat = True
                Length_of_snake += 1

            
            # self.clock.tick(60)

            
            
            if train == True:
                  
                info = [action, game_run, Length_of_snake, input_info, [eat, food, [head_x, head_y], [old_headx, old_heady]], state]
                self.ai.to_pool(info)
                
                if game_run == False:
                    
                    self.ai.learn(self.episode)
                    self.episode += 1
                    game_run = True
                    head_x = self.blocks // 2
                    head_y = self.blocks // 2

                    x_change = 0
                    y_change = 0
                    last_key = None
                    snake_List.clear()
                    Length_of_snake = 1
                    foodx, foody = self.make_food()
                        

            if self.episode % 120 == 0:
                self.ai.algo.save_model()
                self.ai.algo.save_target_model()
                

        # pygame.quit()
        # quit()



    # def reset(self):
    #     snake_move_interval = 50  # 蛇移动的时间间隔（毫秒）
    #     last_move_time = pygame.time.get_ticks()

    #     head_x = self.blocks // 2
    #     head_y = self.blocks // 2

    #     x_change = 0
    #     y_change = 0
    #     last_key = None
    #     snake_List = []
    #     Length_of_snake = 1
    #     foodx, foody = self.make_food()

    #     state = [head_x, head_y, x_change, y_change, foodx, foody] + [x for segment in snake_List for x in segment]
    #     return state

