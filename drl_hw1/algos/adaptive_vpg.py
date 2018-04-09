import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
import drl_hw1.samplers.trajectory_sampler as trajectory_sampler
import drl_hw1.samplers.batch_sampler as batch_sampler

# utility functions
import drl_hw1.utils.process_samples as process_samples
from drl_hw1.utils.logger import DataLog
from drl_hw1.algos.batch_reinforce import BatchREINFORCE

class AdaptiveVPG(BatchREINFORCE):
    def __init__(self, env, policy, baseline,
                 learn_rate=10.0, # alpha (should be picked large enough)
                 kl_desired=0.01, # delta bar (0.01-0.1 is a reasonable range)
                 seed=None,
                 save_logs=False):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = learn_rate
        self.delta = kl_desired
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        if save_logs: self.logger = DataLog()

    # ----------------------------------------------------------
    def train_from_paths(self, paths):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_opt = 0.0

        # Optimization algorithm
        # --------------------------

        # **************************
        # Change the below code to include linesearch
        # **************************

        ts = timer.time()
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        curr_params = self.policy.get_param_values()        
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        alpha = copy.deepcopy(self.alpha)
        new_params, new_surr, kl_dist = self.simple_gradient_update(curr_params, vpg_grad, alpha,
                                    observations, actions, advantages)
        self.policy.set_param_values(new_params, set_new=True, set_old=True)
        surr_improvement = new_surr - surr_before
        t_opt += timer.time() - ts

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', self.alpha)
            self.logger.log_kv('time_opt', t_opt)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_improvement)
            self.logger.log_kv('running_score', self.running_score)

        return base_stats

    def simple_gradient_update(self, curr_params, search_direction, step_size,
                               observations, actions, advantages):
        # This function takes in the current parameters, a search direction, and a step size
        # and computes the new_params =  curr_params + step_size * search_direction.
        # It also computes the CPI surrogate at the new parameter values.
        # This function also computes KL(pi_new || pi_old) as discussed in the class,
        # where pi_old = policy with current parameters (i.e. before any update),
        # and pi_new = policy with parameters equal to the new_params as described above.
        # The function DOES NOT set the parameters to the new_params -- this has to be
        # done explicitly outside this function.

        new_params = curr_params + step_size*search_direction
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        new_surr = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(curr_params, set_new=True, set_old=True)
        return new_params, new_surr, kl_dist
