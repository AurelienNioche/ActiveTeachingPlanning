import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import git

from a2c.a2c import A2C
from a2c.callback.teacher import TeacherCallback

from environments.continuous_teaching import ContinuousTeaching
from environments.discontinuous_teaching import DiscontinuousTeaching
from environments import reward_types
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

BKP_FOLDER = "bkp/single_env"
FIG_FOLDER = "fig/single_env"
os.makedirs(BKP_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)


def run_discontinuous_teaching(reward_type, gamma):

    forgets, repetitions = generate_learners_parameterization(
        n_users=N_USERS, n_items=N_ITEMS, seed=SEED_PARAM_LEARNERS)

    env = DiscontinuousTeaching(
        tau=0.9,
        break_length=24 * 60 ** 2,
        time_per_iter=3,
        n_iter_per_session=100,
        n_session=6,
        initial_forget_rates=forgets,
        initial_repetition_rates=repetitions,
        delta_coeffs=np.array([3, 20]),
        n_item=N_ITEMS,
        penalty_coeff=0.2,
        reward_type=reward_type,
        gamma=gamma
    )
    # layers_dim = [64, 64, 128]
    m = A2C(env,
            normalize_advantage=True,
            # net_arch=[{'pi': layers_dim, 'vf': layers_dim}],
            seed=123
        )

    env_t_max = env.n_session * env.n_iter_per_session
    iterations = env_t_max * 30000
    check_freq = env_t_max

    with TeacherCallback(env, check_freq) as callback:
        m.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.show()
    return m


def run_continuous_teaching(reward_type, gamma=1):

    forgets, repetitions = generate_learners_parameterization(
        n_users=N_USERS, n_items=N_ITEMS, seed=SEED_PARAM_LEARNERS)

    env = ContinuousTeaching(
        t_max=100,
        initial_forget_rates=forgets,
        initial_repetition_rates=repetitions,
        n_item=N_ITEMS,
        tau=0.9,
        delta_coeffs=np.array([3, 20]),
        penalty_coeff=0.2,
        reward_type=reward_type,
        gamma=gamma
    )

    m = A2C(env)

    # iterations = env.t_max * 5e5
    iterations = int(10e6)
    check_freq = env.t_max

    with TeacherCallback(env, check_freq) as callback:
        m.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.savefig('{}.png'.format(gamma))
    plt.show()
    return m


def main():
    gamma = 8
    reward_type = reward_types.EXAM_BASED

    model = run_discontinuous_teaching(
        reward_type=reward_type,
        gamma=gamma)
    model.save(f'{BKP_FOLDER}/continuous_{COMMIT_NAME}_{gamma}.p')


if __name__ == "__main__":
    main()
