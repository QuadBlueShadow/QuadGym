from MatchFile import make_match
from InstanceManager import SingleInstance
from Models import RandomAgentKBM
import os

agents_per_match = 2

#Save data after this many steps
save_marker = 500_000

#Get the correct save directory and instance number for our instance
not_found = True
instance_num = 0
data_save_dir = f"C:/example_data/data{instance_num}.npy"

model_save_dir = f"C:/example_models/main_save.zip"

while not_found:
    if os.path.exists(data_save_dir):
        instance_num += 1

#Start our instance
env = SingleInstance()
env.start_match(make_match, agents_per_match)

#Make Model
model = RandomAgentKBM()

#Run our instance
env.run_match(model, save_marker, data_save_dir)