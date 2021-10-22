from abc import ABC
from typing import Union

import gym
import numpy as np


class TeachingExam(gym.Env, ABC):

    def __init__(
            self,
            learned_threshold=0.9,
            n_item=30,
            n_session: int = 6,                     # 6
            n_iter_per_session: int = 100,          # 100
            break_length: Union[float, int] = 10,   # 24*60**2
            time_per_iter: Union[float, int] = 1,  # 4
            init_forget_rate: float = 0.2,
            rep_effect: float = 0.2
    ):
        super().__init__()

        # Action space
        self.action_space = gym.spaces.Discrete(n_item)

        # Task parameter
        self.n_item = n_item
        self.time_per_iter = time_per_iter
        self.n_session = n_session
        self.n_iter_per_session = n_iter_per_session
        self.break_length = break_length

        # Threshold for reward computation
        self.log_thr = np.log(learned_threshold)

        # Cognitive parameterization
        self.init_forget_rate = np.full(self.n_item, init_forget_rate)
        self.rep_effect = np.full(self.n_item, rep_effect)

        self.total_iteration = self.n_iter_per_session*self.n_session

        # ---------------------------------------------- #

        # Initialize counters
        self.current_total_iter = 0
        self.current_iter = 0
        self.current_ss = 0

        self.time_between = self.time_per_iter

        # Current state of the items
        # column 0: number of iteration elapsed since
        #           the last presentation of the item
        # column 1: number of presentation of the item
        self.item_state = np.zeros((n_item, 2))

        # Current state of memory
        # Column 0: 'seen' => if the item has already been seen or not
        # Column 1: 'survival time' => how long before the item goes under
        #           the threshold (expressed proportionally
        #           to the total number of iteration)
        # Column 2: 'survival time if action' => how long
        #           before the item goes under
        #           the threshold (expressed proportionally
        #           to the total number of iteration)
        #           assuming that it is presented at the next iteration
        self.memory_state = np.zeros((self.n_item, 3))

        # Observation vector as given to the RL agent
        # +1 because we give the progress
        self.obs_dim = n_item * self.memory_state.shape[1] + 1
        self.obs = np.zeros(self.obs_dim)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf,
            shape=(self.obs_dim, ))

        self.reward = 0
        self.done = False

    def reset(self):

        self.item_state[:] = 0.
        self.memory_state[:] = 0.
        self.obs[:] = 0.

        self.current_iter = 0
        self.current_ss = 0
        self.current_total_iter = 0

        self.time_between = self.time_per_iter

        self.reward = 0
        self.done = False

        self.update_obs()

        return self.obs

    def step(self, action):

        self.update_item_state(action)
        self.update_obs()
        self.update_game_state()

        self.update_reward()  # Particular placement

        info = {}
        return self.obs, self.reward, self.done, info

    def update_obs(self):

        seen = self.item_state[:, 1] > 0
        unseen = np.invert(seen)
        delta = self.item_state[seen, 0]  # only consider already seen items
        rep = self.item_state[seen, 1] - 1.  # only consider already seen items

        forget_rate = self.init_forget_rate[seen] * \
            (1 - self.rep_effect[seen]) ** rep

        if self.current_iter == (self.n_iter_per_session - 1):
            # It will be a break before the next iteration
            delta += self.break_length
        else:
            delta += self.time_per_iter

        survival = - (self.log_thr / forget_rate) - delta
        survival[survival < 0] = 0.

        seen_f_rate_if_action = self.init_forget_rate[seen] * \
            (1 - self.rep_effect[seen]) ** (rep + 1)

        seen_survival_if_action = - self.log_thr / seen_f_rate_if_action

        unseen_f_rate_if_action = self.init_forget_rate[unseen]
        unseen_survival_if_action = - self.log_thr / unseen_f_rate_if_action

        self.memory_state[:, 0] = seen
        self.memory_state[seen, 1] = survival
        self.memory_state[unseen, 1] = 0.
        self.memory_state[seen, 2] = seen_survival_if_action
        self.memory_state[unseen, 2] = unseen_survival_if_action

        progress = (self.current_total_iter + 1) / self.total_iteration

        self.memory_state[:, 1:3] /= self.total_iteration

        self.obs[:-1] = self.memory_state.flatten()
        self.obs[-1] = progress

    def update_item_state(self, action):

        self.item_state[:, 0] += self.time_between      # add time elapsed since last iter
        self.item_state[action, 0] = self.time_between  # ...except for item shown
        self.item_state[action, 1] += 1                 # increment nb of presentation

    @classmethod
    def get_p_recall(cls, obs):
        obs = obs.reshape((obs.shape[0] // 2, 2))
        return obs[:, 0]

    def update_game_state(self):

        self.current_iter += 1
        self.current_total_iter += 1
        if self.current_iter >= self.n_iter_per_session:
            self.current_iter = 0
            self.current_ss += 1
            time_between = self.break_length
        else:
            time_between = self.time_per_iter

        if self.current_total_iter == self.total_iteration:
            self. done = True
        else:
            self.done = False

        # for next iteration
        self.time_between = time_between

    def update_reward(self):

        if self.done:

            # self.item_state[:, 0] += self.time_between  # add time elapsed since last iter

            seen = self.item_state[:, 1] > 0
            delta = self.item_state[seen, 0]  # only consider already seen items
            rep = self.item_state[seen, 1] - 1.  # only consider already seen items
            forget_rate = self.init_forget_rate[seen] * \
                (1 - self.rep_effect[seen]) ** rep
            log_p_recall = - forget_rate * delta
            above_thr = log_p_recall > self.log_thr
            n_learned_now = np.count_nonzero(above_thr)
            self.reward = n_learned_now / self.n_item