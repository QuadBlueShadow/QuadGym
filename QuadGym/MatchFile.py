import numpy as np
import random
import rlgym

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
#Fps
fps = 120 / tick_skip
#Number of instances of rocket league
num_instances = 1

#Simple match function
def make_match():
    env = rlgym.make(
        team_size=1,
        tick_skip=tick_skip,
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
        spawn_opponents=True,
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
        obs_builder=AdvancedObs(),
        state_setter=DefaultState(),
        action_parser=DiscreteAction()
    )
    return env
