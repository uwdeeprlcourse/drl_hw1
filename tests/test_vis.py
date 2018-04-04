from drl_hw1.utils.gym_env import GymEnv
from drl_hw1.policies.gaussian_mlp import MLP
from drl_hw1.baselines.linear_baseline import LinearBaseline
from drl_hw1.algos.batch_reinforce import BatchREINFORCE
from drl_hw1.utils.train_agent import train_agent
import drl_hw1.envs
import time as timer
SEED = 500

e = GymEnv('drl_hw1_point_mass-v0')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
baseline = LinearBaseline(e.spec)
agent = BatchREINFORCE(e, policy, baseline, learn_rate=1e-3, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='test_exp1',
            agent=agent,
            seed=SEED,
            niter=10,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=5,
            sample_mode='trajectories',
            num_traj=100,
            save_freq=5,
            evaluation_rollouts=None)
print("time taken = %f" % (timer.time()-ts))
e.visualize_policy(policy, num_episodes=5, horizon=e.horizon, mode='exploration')
