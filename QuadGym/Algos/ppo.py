import torch as t
import numpy as np

class PPO:
    def __init__(self) -> None:
        pass

    def learn(self, train_data, model):
        for episode in train_data:
            for step in episode:
                for obs, reward, action in step:
                    agent_num = len(obs)