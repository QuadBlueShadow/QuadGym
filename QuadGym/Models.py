import random as rand
import numpy as np
import torch as t

class BaseModelR():
    def __init__(self) :
        self.n_inputs = 8

    def get_actions(self, obs):
        #Set up randomly for a discrete action parser
        actions = np.array([])
        actions = np.append(actions, [rand.randint(-1, 1)], axis=0)
        actions = np.append(actions, [rand.randint(-1, 1)], axis=0)
        actions = np.append(actions, [rand.randint(-1, 1)], axis=0)
        actions = np.append(actions, [rand.randint(-1, 1)], axis=0)
        actions = np.append(actions, [rand.randint(-1, 1)], axis=0)

        actions = np.append(actions, [rand.randint(0, 1)], axis=0)
        actions = np.append(actions, [rand.randint(0, 1)], axis=0)

        return actions

class RandomAgentKBM(BaseModelR):
    def __init__(self, obs):
        self.n_inputs = 5

    def get_actions(self):
        #Random actions using KBM action parser
        actions = np.array([])
        actions = np.append(actions, [rand.randint(-1, 1)], axis=0)
        actions = np.append(actions, [rand.randint(-1, 1)], axis=0)
        actions = np.append(actions, [rand.randint(-1, 1)], axis=0)

        actions = np.append(actions, [rand.randint(0, 1)], axis=0)
        actions = np.append(actions, [rand.randint(0, 1)], axis=0)

        return actions