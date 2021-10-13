import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import git

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback

from environments.continuous_teaching import ContinuousTeaching, types
from generate_learners.generate_learners import \
    generate_learners_parameterization

sns.set()

N_USERS = 5
N_ITEMS = 60
SEED_PARAM_LEARNERS = 123

REPO = git.Repo(search_parent_directories=True)
GIT_BRANCH = REPO.active_branch.name
GIT_HASH = REPO.head.commit.hexsha
COMMIT_NAME = GIT_BRANCH + '_' + GIT_HASH

EXPERIMENT_NAME = ''

BKP_FOLDER = "bkp/curriculum_runs"
FIG_FOLDER = "fig/curriculum_runs"
os.makedirs(BKP_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)


def curriculum_learning(reward_type, gamma, session_lengths=(50, 100)):

    forgets, repetitions = generate_learners_parameterization(
        n_users=N_USERS, n_items=N_ITEMS, seed=SEED_PARAM_LEARNERS)

    env = ContinuousTeaching(
        t_max=100,
        initial_forget_rates=forgets,
        initial_repetition_rates=repetitions,
        n_item=30,
        tau=0.9,
        delta_coeffs=np.array([3, 20]),
        penalty_coeff=0.2,
        reward_type=reward_type,
        gamma=gamma
    )
    m = A2C(env)
    iterations = int(10e6 / len(session_lengths))
    check_freq = env.t_max
    for i in range(len(session_lengths)):
        env.t_max = session_lengths[i]
        if i > 0:
            gamma *= session_lengths[i] / session_lengths[i - 1]
        env.gamma = gamma
        with ProgressBarCallback(env, check_freq) as callback:
            m.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.savefig(f'{FIG_FOLDER}/{EXPERIMENT_NAME}_{COMMIT_NAME}_{gamma}.png')
    plt.clf()

    return m


def main():
    for i in [2, 3, 4, 5, 8]:
        print('Running on {}...'.format(i))
        model = curriculum_learning(
            reward_type=types['exam_based'],
            gamma=i,
            session_lengths=(20, 50, 100))
        model.save(f'{BKP_FOLDER}/eb21_{COMMIT_NAME}_{i}.p')


if __name__ == "__main__":
    main()
