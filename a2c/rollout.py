from collections import namedtuple

import torch
import numpy as np


RolloutData = namedtuple("RolloutData",
                         ["observations",
                          "actions",
                          "advantages",
                          "returns"])


class RolloutBuffer:
    """
    Rollout buffer corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.

    Is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param obs_shape: Shape of one observation
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    """

    def __init__(
            self,
            buffer_size: int,
            obs_shape: tuple,
            gae_lambda: float = 1,
            gamma: float = 0.99):
        
        self.buffer_size = buffer_size

        self.obs_shape = obs_shape

        self.pos = 0
        self.full = False

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None

        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, ) + self.obs_shape,
                                     dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros_like(self.actions)
        self.returns = np.zeros_like(self.actions)
        self.dones = np.zeros_like(self.actions)
        self.values = np.zeros_like(self.actions)
        self.log_probs = np.zeros_like(self.actions)
        self.advantages = np.zeros_like(self.actions)

        self.pos = 0
        self.full = False

    def add(
            self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
            done: np.ndarray, value: torch.Tensor, log_prob: torch.Tensor
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param done: End of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_value: torch.Tensor,
                                      done: np.ndarray) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_value: tensor with unique element
        :param done: tensor with unique element

        """
        # convert to numpy
        last_value = last_value.numpy()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self) -> RolloutData:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)
        rollout_data = RolloutData(
            observations=torch.as_tensor(self.observations[indices]),
            actions=torch.as_tensor(self.actions[indices]),
            advantages=torch.as_tensor(self.advantages[indices].flatten()),
            returns=torch.as_tensor(self.returns[indices]))
        return rollout_data
