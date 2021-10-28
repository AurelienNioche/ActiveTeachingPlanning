import os
import torch
from torch import nn
from torch import distributions as dist
from tqdm import tqdm
import numpy as np
import math
import optuna
import functools

from a2c.a2c import A2C
from a2c.callback.teacher_exam import TeacherExamCallback
from environments.teaching_exam import TeachingExam as TeacherEnv
from baseline_policies.leitner import Leitner
from baseline_policies.threshold import Threshold


def run(env, policy, seed=123):

    np.random.seed(seed)

    rewards = []
    actions = []

    obs = env.reset()

    # with tqdm(total=env.n_iter_per_session * env.n_session) as pb:
    while True:
        action = policy.act(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
        # pb.update()
        if done:
            break

    final_n_learned = rewards[-1] * env.n_item
    # n_view = len(np.unique(np.asarray(actions)))
    # print(f"{policy.__class__.__name__.lower()} | "
    #       f"final reward {int(final_n_learned)} | "
    #       f"precision {final_n_learned / n_view:.2f}")
    return final_n_learned


# class Model(nn.Module):
#
#     def __init__(self,
#                  env_kwargs,
#                  teaching_iterations=int(1e4)):
#
#         super(Model, self).__init__()
#
#         self.loc = nn.Parameter(torch.zeros(1))
#         self.log_scale = nn.Parameter(torch.zeros(1))
#
#         self.env_kwargs = env_kwargs
#
#         self.teaching_iterations = teaching_iterations
#
#     def get_action(self):
#
#         action = dist.Normal(loc=self.loc, scale=self.log_scale.exp()).sample((1,))[0, 0]
#         action = math.tanh(action)
#         low, high = -49, 49
#         action = low + (0.5 * (action + 1.0) * (high - low))
#         return 51 + int(round(action))
#
#     def step(self, teacher):
#
#         n_iter_per_session = self.get_action()
#
#         teacher.env = TeacherEnv(n_iter_per_session=n_iter_per_session,
#                                  **self.env_kwargs)
#         teacher.buffer_size = teacher.env.n_iter_per_session * teacher.env.n_session
#         print("begin learning")
#         teacher.learn(self.teaching_iterations, callback=None)
#         print("yeah")
#
#         return run(teacher.env, teacher)


def objective(trial, teacher, env_kwargs, teaching_iterations=int(1e4)):

    n_iter_per_session = trial.suggest_int('n_iter_per_session', 1, 100)
    teacher.env = TeacherEnv(n_iter_per_session=n_iter_per_session,
                             **env_kwargs)
    teacher.buffer_size = teacher.env.n_iter_per_session * teacher.env.n_session

    teacher.learn(teaching_iterations, callback=None)

    return run(teacher.env, teacher)


def teach_the_teacher(epochs=1000):
    env_kwargs = dict(
        init_forget_rate=0.02,
        rep_effect=0.2,
        n_item=30,
        learned_threshold=0.9,
        n_session=1,
        time_per_iter=1,
        break_length=1
    )
    env = TeacherEnv(n_iter_per_session=10, **env_kwargs)
    teacher = A2C(env=env)

    bkp_file = "bkp/optuna.db"
    if os.path.exists(bkp_file):
        os.remove(bkp_file)

    study = optuna.create_study(direction='maximize',
                                storage=f"sqlite:///{bkp_file}",
                                study_name="optuna_teach_the_teacher")
    study.optimize(functools.partial(objective, teacher=teacher, env_kwargs=env_kwargs),
                   n_trials=epochs)
#
# class Model(nn.Module):
#
#     def __init__(self,
#                  env_kwargs,
#                  teaching_iterations=int(1e6)):
#
#         super(Model, self).__init__()
#
#         self.loc = nn.Parameter(torch.zeros(1))
#         self.log_scale = nn.Parameter(torch.zeros(1))
#
#         self.env_kwargs = env_kwargs
#
#         self.teaching_iterations = teaching_iterations
#
#     def get_action(self):
#
#         action = dist.Normal(loc=self.loc, scale=self.log_scale.exp()).sample((1,))[0, 0]
#         action = math.tanh(action)
#         low, high = -49, 49
#         action = low + (0.5 * (action + 1.0) * (high - low))
#         return 51 + int(round(action))
#
#     def step(self, teacher):
#
#         n_iter_per_session = self.get_action()
#
#         teacher.env = TeacherEnv(n_iter_per_session=n_iter_per_session,
#                                  **self.env_kwargs)
#         teacher.buffer_size = teacher.env.n_iter_per_session * teacher.env.n_session
#
#         teacher.learn(self.teaching_iterations, callback=None)
#
#         return run(teacher.env, teacher)
#
#
# def teach_the_teacher(epochs=1000, learning_rate=0.01):
#
#     constant_lr = True
#
#     env_kwargs = dict(
#         init_forget_rate=0.02,
#         rep_effect=0.2,
#         n_item=30,
#         learned_threshold=0.9,
#         n_session=1,
#         time_per_iter=1,
#         break_length=1
#     )
#     env = TeacherEnv(n_iter_per_session=10, **env_kwargs)
#     teacher = A2C(env=env)
#
#     model = Model(env_kwargs=env_kwargs)
#
#     optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate)
#
#     with tqdm(total=epochs) as pb:
#         # Maybe maintain constant the learning rate
#         if constant_lr:
#             for param_group in optimizer.param_groups:
#                 param_group["lr"] = learning_rate
#
#         loss = - model.step(teacher=teacher)
#
#         # Optimization step
#         optimizer.zero_grad()
#         loss.backward()
#
#         # # Clip grad norm
#         # nn.utils.clip_grad_norm_(model.parameters(),
#         #                          self.max_grad_norm)
#         optimizer.step()
#
#         pb.update()


if __name__ == "__main__":
    teach_the_teacher()