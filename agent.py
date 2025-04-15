import queue
import algorithm
import random
import torch

class AGENT():
    def __init__(self, inputsize):
        if torch.cuda.is_available():
            self.algo = algorithm.ALGO('gpu', inputsize, 4)
            print("use gpu!")
        else:
            print("使用CPU进行训练！")
            self.algo = algorithm.ALGO('cpu', inputsize, 4)
        
        self.episodes = 1000
        self.max_size = 100
        self.batch_size = 20
        self.buffer = queue.deque(self.max_size)
    def to_pool(self, info):
        if len(self.buffer) == self.max_size:
            self.buffer.pop()
        self.buffer.push(info)

    def predict(self, obs):
        return self.algo.predictQ(obs)

    def learn(self):
        '''
        每隔两百次同步一下，target和model的参数
        '''
        
        if self.episodes % 200 == 0:
            self.algo.TargetQ = self.algo.Qvalue
        
        batch = random.sample(self.buffer, self.batch_size)
        for data in batch:
            self.algo.learn(data)

