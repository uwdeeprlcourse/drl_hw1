from gym.envs.registration import register

# ----------------------------------------
# drl_hw1 environments
# ----------------------------------------

register(
    id='drl_hw1_point_mass-v0',
    entry_point='drl_hw1.envs:PointMassEnv',
    max_episode_steps=25,
)

register(
    id='drl_hw1_swimmer-v0',
    entry_point='drl_hw1.envs:SwimmerEnv',
    max_episode_steps=500,
)

from drl_hw1.envs.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from drl_hw1.envs.point_mass import PointMassEnv
from drl_hw1.envs.swimmer import SwimmerEnv