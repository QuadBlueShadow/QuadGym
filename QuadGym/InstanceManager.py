import numpy as np
import time
from Models import BaseModelR
import os
from Algos.ppo import PPO
import torch as t

class SingleInstance():
    def __init__(self) -> None:
        self.env = None
        self.team_size = 1
        self.data = []
        self.episode = []

    def start_match(self, match_fun, agents_per_match):
        #Start the match
        self.agents_per_match = agents_per_match

        self.env = match_fun()
        time.sleep(2)
        print("Instance Done")
        print(" ")

    def step(self, actions):
        #Return all of the goodies for training
        next_obs, reward, done, gameinfo = self.env.step(actions)
        return next_obs, reward, done, gameinfo

    def run_match(self, model:BaseModelR, save_marker, data_save_dir):
        #Make sure to set our env
        obs = self.env.reset()
        done = False
        steps = 0
        
        while True:
            #Take an action
            all_actions = []
            for i in range(self.agents_per_match):
                all_actions.append(model.get_actions(obs))

            #Convert to numpy array
            all_actions = np.asarray(all_actions)

            #Take a step and grab the info
            next_obs, reward, done, gameinfo = self.step(all_actions)
            obs = next_obs

            #Put the data in an array
            self.episode.append(next_obs, reward, all_actions)
            if done:
                #Put our episdoe into the data array and clear episode
                self.data.append(self.episode)
                self.episode = []

                #Reset the env if timeout condition
                self.env.reset()
                done = False

            #Save our data once we have reached batch size
            if steps > save_marker:
                save_data = np.asarray(self.data)
                #Try to combine old info with new info into a single info file
                try:
                    old_data = np.load(data_save_dir)

                    save_data = np.concatenate((old_data, save_data), axis=0)
                except:
                    pass

                #Save array
                np.save(data_save_dir, save_data)
                self.data = []
                steps = 0

            steps += 1

class LearnerInstance():
    def __init__(self, check_timer=5, lr=0.005, n_epochs=1):
        self.check_timer = check_timer*60
        self.lr = lr
        self.n_epochs = n_epochs

    def get_filenames(self, dir):
        filenames = os.listdir(dir)
        return filenames

    def get_data(self, save_dir):
        fn = self.get_filenames(save_dir)
        data = None

        #Put all of our data into one numpy array
        for i in range(len(fn)):
            if type(data) == None:
                data = np.load(save_dir+fn[i])
            else:
                new_data = np.load(save_dir+fn[i])
                data = np.concatenate((data, new_data), axis=0)

        return data

    def get_model(self, dir):
        model = t.load(dir)
        return model

    def run(self, model_save_dir, data_save_dir):
        model = self.get_model(model_save_dir)
        while True:
            train_data = self.get_data(data_save_dir)

            for i in range(self.n_epochs):
                PPO.learn(train_data, model)

            time.sleep(self.check_timer)