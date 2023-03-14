import numpy as np
import random
import rlgym

from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition

from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward

from rlgym.utils import RewardFunction
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z, CAR_MAX_SPEED, BALL_RADIUS
from rlgym.utils.gamestates import GameState, PlayerData

from rlgym.utils.state_setters import DefaultState, RandomState
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper

#Normal Tick Skip
tick_skip = 8
#1v1 Match
agents_per_match = 2
#Fps
fps = 120 / tick_skip
#Number of instances of rocket league
num_instances = 1

class AirStateSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
    
        # Set up our desired spawn location and orientation. Here, we will only change the yaw, leaving the remaining orientation values unchanged.
        desired_yaw = np.pi/2
        
        # Loop over every car in the game.
        for car in state_wrapper.cars:
            desired_car_pos = [random.randint(-500,500),random.randint(-500,500),17] #x, y, z
            if car.team_num == BLUE_TEAM:
                pos = desired_car_pos
                yaw = 0
                
            elif car.team_num == ORANGE_TEAM:
                # We will invert values for the orange team so our state setter treats both teams in the same way.
                pos = [-1*coord for coord in desired_car_pos]
                yaw = 0
                
            # Now we just use the provided setters in the CarWrapper we are manipulating to set its state. Note that here we are unpacking the pos array to set the position of 
            # the car. This is merely for convenience, and we will set the x,y,z coordinates directly when we set the state of the ball in a moment.
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33
            
        # Now we will spawn the ball in the center of the field, floating in the air.
        state_wrapper.ball.set_pos(x=random.randint(-500,500), y=random.randint(-500,500), z=CEILING_Z/1.1)
        state_wrapper.ball.set_lin_vel(x=random.randint(-1500, 1500), y=random.randint(-1500, 1500), z=random.randint(-1500, 1500))
        state_wrapper.ball.set_ang_vel(x=random.randint(-100,100), y=random.randint(-100,100), z=random.randint(-100,100))

class MasterStateSetter(StateSetter):
    #Choose a random state setter to use
    def __init__(
            self, *,
            random_prob=0.20,
            kickoff_prob=0.45,
            air_prob=0.35,
    ):

        super().__init__()

        self.setters = [
            RandomState(),
            DefaultState(),
            AirStateSetter(),
        ]
        self.probs = np.array([random_prob, kickoff_prob, air_prob])
        assert self.probs.sum() == 1, "Probabilities must sum to 1"

    def reset(self, state_wrapper: StateWrapper):
        i = np.random.choice(len(self.setters), p=self.probs)
        self.setters[i].reset(state_wrapper)

class JumpTouchReward(RewardFunction):
        def __init__(self, min_height=130, exp=0.31):
            self.min_height = min_height
            self.exp = exp

        def reset(self, initial_state: GameState):
            pass
    
        def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
        ) -> float:
            if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
                return ((state.ball.position[2] - 92) ** self.exp)-1

            return 0
    
class TeamSpacingReward(RewardFunction):
        def __init__(self):
            self.team_size = agents_per_match/2

        def reset(self, initial_state: GameState):
            pass
    
        def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
        ) -> float:
            reward = 0
            desired_dist = 500
            if self.team_size > 2:
                dist_1 = 0
                dist_2 = 0
                dist_3 = 0
                if player.team_num == BLUE_TEAM:
                    dist_1 = np.linalg.norm(state.players[0].car_data.position - state.players[1].car_data.position)
                    dist_2 = np.linalg.norm(state.players[1].car_data.position - state.players[2].car_data.position)
                    dist_3 = np.linalg.norm(state.players[0].car_data.position - state.players[2].car_data.position)
                else:
                    dist_1 = np.linalg.norm(state.players[3].car_data.position - state.players[4].car_data.position)
                    dist_2 = np.linalg.norm(state.players[4].car_data.position - state.players[5].car_data.position)
                    dist_3 = np.linalg.norm(state.players[3].car_data.position - state.players[5].car_data.position)

                if dist_1 > desired_dist:
                    reward += 0.33
                else:
                    reward = reward-0.25

                if dist_2 > desired_dist:
                    reward += 0.33
                else:
                    reward -= 0.25

                if dist_3 > desired_dist:
                    reward += 0.33
                else:
                    reward -= 0.25
            elif self.team_size == 2:
                dist = 0
                if player.team_num == BLUE_TEAM:
                    dist = np.linalg.norm(state.players[0].car_data.position - state.players[1].car_data.position)
                else:
                    dist = np.linalg.norm(state.players[2].car_data.position - state.players[3].car_data.position)

                if dist > desired_dist:
                    reward = 1
                else:
                    reward = -0.5
                
            return reward

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
            #JumpTouchReward(),
            #ThirdManReward(),
            #TeamSpacingReward(),
            #RetreatSpeedReward(),
            #SpeedReward(),
            #TerminalVelocityBoostConserveReward(),
            #JumpshotHieghtReward(),
            #BallOnOtherHalfReward(),
        ),
        (0.1, 1.0, 1.0)),
        spawn_opponents=True,
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
        obs_builder=AdvancedObsPadder(),
        state_setter=MasterStateSetter(),
        action_parser=KBMAction()
    )
    return env