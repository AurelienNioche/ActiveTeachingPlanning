from typing import Union
from abc import ABC

import gym
import numpy as np

from .reward_types import types


class DiscontinuousTeaching(gym.Env, ABC):

    def __init__(
            self,                              # Setting previous XP
            initial_forget_rates: np.ndarray,
            initial_repetition_rates: np.ndarray,
            delta_coeffs: np.array,
            penalty_coeff: float = 0.2,
            tau: float = 0.9,
            n_item: int = 30,                       # 500
            n_session: int = 6,                     # 6
            n_iter_per_session: int = 100,          # 100
            break_length: Union[float, int] = 10,   # 24*60**2
            time_per_iter: Union[float, int] = 1,  # 4
            reward_coeff: float = 1,
            reward_type=types['monotonic'],
            gamma=1
    ):
        super().__init__()

        self.action_space = gym.spaces.Discrete(n_item)
        self.n_item = n_item

        self.tau = tau
        self.log_tau = np.log(self.tau)

        n_coeffs = delta_coeffs.shape[0]
        if n_coeffs < 1:
            raise ValueError(
                "The length of delta_coeffs should be superior or equal to 1"
            )
        self.delta_coeffs = delta_coeffs
        self.obs_dim = n_coeffs + 2
        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf,
                                                shape=(n_item * self.obs_dim + 1,))
        self.learned_before = np.zeros((self.n_item, ))
        self.penalty_coeff = penalty_coeff

        self.n_users = initial_forget_rates.shape[0]
        self.current_user = self.pick_a_user()
        if initial_repetition_rates.shape[1] == n_item and \
                initial_forget_rates.shape[1] == n_item:
            self.all_forget_rates = initial_forget_rates
            self.all_repetition_rates = initial_repetition_rates
            self.initial_forget_rates = initial_forget_rates[self.current_user]
            self.initial_repetition_rates = initial_repetition_rates[self.current_user]
        else:
            raise ValueError(
                "Mismatch between initial_rates shapes and n_item"
            )

        self.n_session = n_session
        self.n_iter_per_session = n_iter_per_session
        self.t_max = n_session * n_iter_per_session
        self.break_length = break_length
        self.time_per_iter = time_per_iter
        self.reward_coeff = reward_coeff
        self.reward_type = reward_type

        # Things that need to be reset
        self.state = np.zeros((n_item, 2))
        self.obs = np.zeros((n_item, self.obs_dim))
        self.current_iter = 0
        self.current_ss = 0
        self.time_elapsed_since_last_iter = 0
        self.gamma = gamma

    def pick_a_user(self):
        self.current_user = np.random.randint(self.n_users)
        return self.current_user

    def reset(self, user=None):
        if user is None:
            user = self.pick_a_user()
        self.current_user = user
        self.state = np.zeros((self.n_item, 2))
        self.initial_forget_rates = self.all_forget_rates[self.current_user]
        self.initial_repetition_rates = self.all_repetition_rates[self.current_user]
        self.obs = np.zeros((self.n_item, self.obs_dim))
        self.learned_before = np.zeros((self.n_item, ))
        self.obs[:, 2] = self.initial_repetition_rates
        self.obs[:, 3] = self.initial_forget_rates
        self.current_iter = 0
        self.current_ss = 0
        self.time_elapsed_since_last_iter = 0
        return self.format_obs(0.)

    def format_obs(self, session_progression):
        return np.hstack(
            (self.obs.flatten(), np.array([session_progression, ])))

    def compute_reward(self, logp_recall):
        above_thr = logp_recall > self.log_tau
        n_learned_now = np.count_nonzero(above_thr)
        # penalizing_factor = n_learned_now - np.count_nonzero(self.learned_before)
        # if n_learned_now > 0:
        #     penalizing_factor /= n_learned_now

        if self.reward_type == types['monotonic']:
            learned_diff = n_learned_now - np.count_nonzero(self.learned_before)
            if learned_diff > 0:
                reward = n_learned_now / (self.gamma * self.n_iter_per_session)
            else:
                reward = learned_diff

        elif self.reward_type == types['mean_learned']:
            penalizing_factor = n_learned_now - np.count_nonzero(self.learned_before)
            # penalizing_factor /= n_learned_now

            reward = (1 - self.penalty_coeff) * (np.count_nonzero(above_thr) / self.n_item) \
                     + self.penalty_coeff * penalizing_factor

        elif self.reward_type == types['exam_based']:
            reward = 10 ** (n_learned_now / self.n_item)

        elif self.reward_type == types['base']:
            reward = n_learned_now / self.n_item

        elif self.reward_type == types['avoid_forget']:
            session_progression = self.session_progression()
            reward = n_learned_now / self.n_item
            if session_progression == 0:
                learned_diff = n_learned_now - np.count_nonzero(self.learned_before)
                reward += min(learned_diff, 0) * self.gamma

        reward *= self.reward_coeff

        self.learned_before = above_thr
        return reward

    def step(self, action):

        # increase delta for all items
        self.state[:, 0] += self.time_elapsed_since_last_iter
        if action is not None:
            # ...except for item shown
            self.state[action, 0] = 0
            # increment number of presentation
            self.state[action, 1] += 1

        view = self.state[:, 1] > 0
        delta = self.state[view, 0]
        rep = self.state[view, 1] - 1.

        forget_rate = self.initial_forget_rates[view] * \
              (1 - self.initial_repetition_rates[view]) ** rep
        logp_recall = - forget_rate * delta
        reward = self.compute_reward(logp_recall)

        time_before_next_iter, done = self.next_delta()
        # Probability of recall at the time of the next action
        for i in range(self.delta_coeffs.shape[0]):
            self.obs[view, i] = np.exp(
                -forget_rate *
                (self.delta_coeffs[i] * delta + time_before_next_iter)
            )

        # update for next call
        self.time_elapsed_since_last_iter = time_before_next_iter

        # Get session progression at the time of the next action
        session_progression = self.session_progression()

        info = {}
        return self.format_obs(session_progression), reward, done, info

    @classmethod
    def extract_p_recall(cls, obs):
        obs = obs[:-1].reshape(((obs.shape[0] - 1) // 2, 2))
        return obs[:, 0]

    def next_delta(self):

        done = False

        self.current_iter += 1
        if self.current_iter >= self.n_iter_per_session:
            self.current_iter = 0
            self.current_ss += 1
            delta = self.break_length
        else:
            delta = self.time_per_iter

        if self.current_ss >= self.n_session:
            done = True

        return delta, done

    def session_progression(self):
        progress = self.current_iter / (self.n_iter_per_session - 1)
        return progress
