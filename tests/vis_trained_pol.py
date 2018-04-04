from drl_hw1.utils.gym_env import GymEnv
import pickle
import drl_hw1.envs

# point mass
e = GymEnv('drl_hw1_point_mass-v0')
policy = pickle.load(open('point_mass_pol.pickle', 'rb'))
e.visualize_policy(policy, num_episodes=10, horizon=e.horizon, mode='evaluation')
del(e)

# swimmer
e = GymEnv('drl_hw1_swimmer-v0')
policy = pickle.load(open('swimmer_pol.pickle', 'rb'))
e.visualize_policy(policy, num_episodes=5, horizon=e.horizon, mode='exploration')
del(e)