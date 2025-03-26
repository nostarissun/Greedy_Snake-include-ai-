import pygame
import time
import random

class Game:
    pygame.init()

    def __init__(self):
        
        # 定义颜色
        self.white = (255, 255, 255)
        self.yellow = (255, 255, 102)
        self.black = (0, 0, 0)
        self.red = (213, 50, 80)
        self.green = (0, 255, 0)
        self.blue = (50, 153, 213)
        self.light_gray = (200, 200, 200)
        # 定义游戏窗口尺寸
        self.blocks = 40
        self.block = 20
        self.win_width = self.blocks * self.block + 1 + self.blocks
        self.win_height = self.blocks * self.block + 1 + self.blocks
        self.line_width = 1
        # 创建游戏窗口
        self.win = pygame.display.set_mode((self.win_width, self.win_height))
        pygame.display.set_caption('ai贪吃蛇')

        self.clock = pygame.time.Clock()

        self.font_style = pygame.font.SysFont("bahnschrift", 25)
        self.score_font = pygame.font.SysFont("comicsansms", 35)


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


    def gameLoop(self):
        game_run = True

        snake_move_interval = 70  # 蛇移动的时间间隔（毫秒）
        last_move_time = pygame.time.get_ticks()

        head_x = self.blocks // 2
        head_y = self.blocks // 2

        x_change = 0
        y_change = 0
        last_key = None
        snake_List = []
        Length_of_snake = 1
        foodx, foody = self.make_food()

        while game_run:
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

            current_time = pygame.time.get_ticks()
            if current_time - last_move_time >= snake_move_interval:
                # 检测蛇是否超出边界
                new_head_x = head_x + x_change
                new_head_y = head_y + y_change
                new_head_x_pixel = new_head_x * (self.block + self.line_width) + self.line_width
                new_head_y_pixel = new_head_y * (self.block + self.line_width) + self.line_width
                if new_head_x_pixel >= self.win_width or new_head_x_pixel < 0 or new_head_y_pixel >= self.win_height or new_head_y_pixel < 0:
                    game_run = False

                # 检测蛇是否碰到自己的身体
                snake_Head = [new_head_x, new_head_y]
                snake_List.append(snake_Head)

                if len(snake_List) > Length_of_snake:
                    del snake_List[0]
                # if any(segment == snake_Head for segment in snake_List[:-1]):
                #     game_run = False
                for i in range(Length_of_snake - 1):
                    if snake_Head[0] == snake_List[i][0] and snake_Head[1] == snake_List[i][1]:
                        game_run = False

                head_x = new_head_x
                head_y = new_head_y
                last_move_time = current_time

            self.main_page()
            pygame.draw.rect(self.win, self.green, [foodx, foody, self.block, self.block])

            self.show_snake(self.block, snake_List, Length_of_snake)
            self.print_score(Length_of_snake - 1)

            pygame.display.update()

        
            head_x_pixel = head_x * (self.block + self.line_width) + self.line_width
            head_y_pixel = head_y * (self.block + self.line_width) + self.line_width
            if head_x_pixel == foodx and head_y_pixel == foody:
                foodx, foody = self.make_food()
                Length_of_snake += 1

            self.clock.tick(60)
        # message("LOSE", red)
        pygame.quit()
        quit()


