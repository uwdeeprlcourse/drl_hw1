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


class BatchREINFORCE:
    def __init__(self, env, policy, baseline,
                 learn_rate=0.01,
                 seed=None,
                 save_logs=False):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        if save_logs: self.logger = DataLog()

    def CPI_surrogate(self, observations, actions, advantages):
        advantages = advantages / (np.max(advantages) + 1e-8)
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR*adv_var)
        return surr

    def kl_old_new(self, observations, actions):
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, observations, actions, advantages):
        cpi_surr = self.CPI_surrogate(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy.trainable_params)
        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad

    # ----------------------------------------------------------
    def train_step(self, N,
                   sample_mode='trajectories',
                   env_name=None,
                   T=1e6,
                   gamma=0.995,
                   gae_lambda=0.98,
                   num_cpu='max'):

        # Clean up input arguments
        if env_name is None: env_name = self.env.env_id
        if sample_mode is not 'trajectories' and sample_mode is not 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()

        if sample_mode is 'trajectories':
            paths = trajectory_sampler.sample_paths_parallel(N, self.policy, T, env_name,
                                                             self.seed, num_cpu)
        elif sample_mode is 'samples':
            paths = batch_sampler.sample_paths(N, self.policy, T, env_name=env_name,
                                               pegasus_seed=self.seed, num_cpu=num_cpu)

        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths
        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)
        # fit baseline
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(paths, return_errors=True)
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(paths)

        return eval_statistics

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
        ts = timer.time()
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        curr_params = self.policy.get_param_values()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        new_params, new_surr, kl_dist = self.simple_gradient_update(curr_params, vpg_grad, self.alpha,
                                        observations, actions, advantages)
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

    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log_kv('stoc_pol_mean', mean_return)
        self.logger.log_kv('stoc_pol_std', std_return)
        self.logger.log_kv('stoc_pol_max', max_return)
        self.logger.log_kv('stoc_pol_min', min_return)

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