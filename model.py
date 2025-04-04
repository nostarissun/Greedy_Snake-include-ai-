import game
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# 定义模型
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 定义Q学习智能体类
#discount_factor（折扣因子）
#用于衡量未来奖励的重要性。它的值通常在 0 到 1 之间，越接近 1，表示未来的奖励越重要；越接近 0，表示智能体更关注当前的奖励
class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}

    def get_q_value(self, state, action):
        # 如果状态不在Q表中，初始化该状态下所有动作的Q值为0
        if state not in self.q_table:
            self.q_table[state] = [0] * self.action_space
        # 返回指定状态和动作对应的Q值
        return self.q_table[state][action]

    def update_q_value(self, state, action, reward, next_state):
        # 计算下一状态下所有动作的最大Q值
        max_q_next = max([self.get_q_value(next_state, i) for i in range(self.action_space)])
        # 获取当前状态和动作的Q值
        current_q = self.get_q_value(state, action)
        # 根据Q学习更新公式更新Q值
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_q_next - current_q)
        # 更新Q表中对应状态和动作的Q值
        self.q_table[state][action] = new_q

    def choose_action(self, state):
        # 以探索率的概率随机选择一个动作进行探索
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.action_space - 1)
        else:
            # 获取当前状态对应的Q值列表
            q_values = self.q_table.get(state, [0] * self.action_space)
            # 选择Q值最大的动作
            return np.argmax(q_values)


def show_model(model, optimizer):
    torch.save(model.state_dict(), 'model.pth')
    model.load_state_dict(torch.load('model.pth'))
    torch.save(optimizer.state_dict(), 'optimizer.pth')
    optimizer.load_state_dict(torch.load('optimizer.pth'))


# 训练AI
def train(use_cpu=True, first_lr=0.001, episodes=100):
    snake_game = game.Game()
    input_size = 7  # 状态空间维度
    output_size = 4  # 动作空间维度
    model = Net(input_size, output_size)
    device = "cpu"
    # 检查是否有可用的GPU
    if not use_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("使用CPU进行训练！")

    try:
        model.load_state_dict(torch.load('model.pth'))
        print("成功加载已有模型！")
    except:
        print("未检测到已有模型！将重新训练")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=first_lr)
    loss_fn = nn.SmoothL1Loss()

    for episode in range(episodes):
        state = snake_game.r
        done = False
        score = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
            next_state, reward, done = snake_game.step(action)
            score += reward

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            next_q_values = model(next_state_tensor).detach()
            max_next_q = torch.max(next_q_values).item()

            target_q = reward + 0.9 * max_next_q
            current_q = q_values[0][action]

            loss = loss_fn(current_q, torch.tensor(target_q).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        print(f"Episode {episode + 1}: Score = {score}")
        torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    train()

    