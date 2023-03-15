from Envs.Match import make_match, Match
from InstanceManager import SingleInstance
from Models import RandomAgentKBM
import os

from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition

from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward

from rlgym.utils.state_setters import DefaultState


#Normal Tick Skip
tick_skip = 8
#1v1 Match
agents_per_match = 2
team_size = 1
#Fps
fps = 120 / tick_skip

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


match = Match(
    reward_function=CombinedReward(
        (
            VelocityPlayerToBallReward(),
            VelocityBallToGoalReward(),
            EventReward(
                team_goal=135.0,
                concede=-155.0,
                shot=10.0,
                save=65.0,
                demo=10.0,
                boost_pickup=1.0,
            ),
        ),
        (0.1, 1.0, 1.0)),
    obs_builder=AdvancedObs(),
    action_parser=DiscreteAction(),
    state_setter=DefaultState(),
    terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
    team_size=team_size,
    tick_skip=tick_skip,
    spawn_opponents=True
)

#Start our instance
env = SingleInstance()
env.start_match(make_match(match), agents_per_match)

#Make Model
model = RandomAgentKBM()

#Run our instance
env.run_match(model, save_marker, data_save_dir)