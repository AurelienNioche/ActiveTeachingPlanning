from abc import ABC

import gym
import numpy as np

from environments.teaching_exam import TeachingExam as TeacherEnv
from a2c.a2c import A2C
import math
# from a2c.callback import SimpleProgressBarCallback


class SupervisorEnv(gym.Env, ABC):

    def __init__(
            self,
            teaching_iterations=int(1e6),
            teaching_n_steps=None,
            n_iter_per_session=10,
            init_forget_rate=0.02,
            rep_effect=0.2,
            n_item=30,
            learned_threshold=0.9,
            n_session=1,
            time_per_iter=1,
            break_length=1
    ):
        super().__init__()

        # Action space
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, ))

        self.teacher_env_kwargs = dict(
            init_forget_rate=init_forget_rate,
            rep_effect=rep_effect,
            n_item=n_item,
            learned_threshold=learned_threshold,
            n_session=n_session,
            time_per_iter=time_per_iter,
            break_length=break_length)

        self.teacher_env = TeacherEnv(n_iter_per_session=n_iter_per_session,
                                      **self.teacher_env_kwargs)

        if teaching_n_steps is None:
            teaching_n_steps = n_iter_per_session*n_session
        self.teacher = A2C(env=self.teacher_env,
                           n_steps=teaching_n_steps)

        obs_dim = len(self.get_teacher_parameters())
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim, ))

        self.teaching_iterations = teaching_iterations

        self.step_counter = 0
        self.reward = -1
        self.n_iter = -1

    def reset(self):

        return self.get_teacher_parameters()

    @staticmethod
    def get_n_iter(action):
        action = action[0, 0]
        action = math.tanh(action)
        low, high = -49, 49
        action = low + (0.5 * (action + 1.0) * (high - low))
        return 51 + int(round(action))

    def step(self, action):

        # Alter environment
        self.n_iter = self.get_n_iter(action)

        self.teacher_env = TeacherEnv(n_iter_per_session=self.n_iter,
                                      **self.teacher_env_kwargs)
        self.teacher.env = self.teacher_env
        n_steps = self.teacher_env.n_iter_per_session * self.teacher_env.n_session
        self.teacher.n_steps = n_steps
        self.teacher.rollout_buffer.buffer_size = n_steps

        self.train_teacher()

        self.reward = self.eval_teacher()

        obs = self.get_teacher_parameters()

        self.step_counter += 1

        done = False
        info = {}
        return obs, self.reward, done, info

    def eval_teacher(self):

        rewards = []

        obs = self.teacher_env.reset()

        while True:
            action = self.teacher.act(obs)
            obs, reward, done, _ = self.teacher_env.step(action)
            rewards.append(reward)

            if done:
                break

        n_learned = rewards[-1] * self.teacher_env.n_item
        return n_learned

    def get_teacher_parameters(self):

        a = self.teacher
        return np.concatenate([p.data.detach().numpy().flatten()
                               for p in a.parameters() if a.requires_grad_()])

    def train_teacher(self):

        # print("TRAIN", "*" * 20)

        self.teacher.learn(self.teaching_iterations, callback=None)

        # print("END TRAIN", "*" * 20)

