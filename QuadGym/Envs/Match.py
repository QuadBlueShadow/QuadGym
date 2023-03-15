import numpy as np
import random
import rlgym

# Match function to feed the config into the make_match function
class Match():
    def __init__(self, 
                 reward_function,
                 obs_builder,
                 action_parser,
                 state_setter,
                 terminal_conditions,
                 team_size=1,
                 tick_skip=8,
                 game_speed=100,
                 gravity=1,
                 boost_consumption=1,
                 spawn_opponents=False):
        super().__init__()

        self.reward_function = reward_function
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.state_setter = state_setter
        self.terminal_conditions = terminal_conditions
        self.team_size = team_size
        self.tick_skip = tick_skip
        self.game_speed = game_speed
        self.gravity = gravity
        self.boost_consumption = boost_consumption
        self.spawn_opponents = spawn_opponents


#Simple match function
def make_match(match: Match):
    env = rlgym.make(
        team_size=match.team_size,
        tick_skip=match.tick_skip,
        reward_function=match.reward_function,
        spawn_opponents=match.spawn_opponents,
        terminal_conditions=match.terminal_conditions,
        obs_builder=match.obs_builder,
        state_setter=match.state_setter,
        action_parser=match.action_parser
    )
    return env


        