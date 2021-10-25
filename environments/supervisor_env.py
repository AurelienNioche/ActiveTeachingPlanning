from abc import ABC

import gym
import numpy as np

from environments.teaching_exam import TeachingExam as TeacherEnv
from a2c.a2c import A2C
from a2c.callback import SimpleProgressBarCallback


class SupervisorEnv(gym.Env, ABC):

    def __init__(
            self,
            teaching_iterations=int(1e6),
            n_action=99,
            init_forget_rate=0.02,
            rep_effect=0.2,
            n_item=30,
            learned_threshold=0.9,
            n_session=1,
            n_iter_per_session=10,
            time_per_iter=1,
            break_length=1
    ):
        super().__init__()

        # Action space
        self.action_space = gym.spaces.Discrete(n_action)

        self.teacher_env = TeacherEnv(
            init_forget_rate=init_forget_rate,
            rep_effect=rep_effect,
            n_item=n_item,
            learned_threshold=learned_threshold,
            n_session=n_session,
            n_iter_per_session=n_iter_per_session,
            time_per_iter=time_per_iter,
            break_length=break_length)

        self.teacher = A2C(env=self.teacher_env)

        obs_dim = len(self.get_teacher_parameters())
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim, ))

        self.teaching_iterations = teaching_iterations

        self.step_counter = 0

    def reset(self):

        return self.get_teacher_parameters()

    def step(self, action):

        # Alter environment
        n_iter = 1 + action
        self.teacher_env.n_iter_per_session = n_iter
        self.train_teacher()

        reward = self.eval_teacher()

        obs = self.get_teacher_parameters()

        self.step_counter += 1
        print(f"Step {self.step_counter} | n_iter={n_iter} | reward={reward}")

        done = False
        info = {}
        return obs, reward, done, info

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

        with SimpleProgressBarCallback() as callback:
            self.teacher.learn(self.teaching_iterations, callback=callback)


