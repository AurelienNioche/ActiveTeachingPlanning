import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from a2c.a2c import A2C
from environments.continuous_teaching import ContinuousTeaching

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback

from environments.continuous_teaching import ContinuousTeaching, types
from environments.discontinuous_teaching import DiscontinuousTeaching
from human_agents import generate_agents

sns.set()
n_users = 5
n_items = 60

LOAD_RATES = True
COMMIT_NAME = '20_50_100'
EXPERIMENT_NAME = ''


def produce_rates():
    global n_items, n_users
    forget_rates, repetition_rates = generate_agents(n_users, n_items)
    print("forget", forget_rates.mean())
    print("repeat", repetition_rates.mean())
    return forget_rates, repetition_rates


def run_discontinuous_teaching(reward_type, gamma):
    global n_items, forgets, repetitions

    # forget_rates, repetition_rates = produce_rates()
    env = DiscontinuousTeaching(
        tau=0.9,
        break_length=24 * 60 ** 2,
        time_per_iter=3,
        n_iter_per_session=100,
        n_session=6,
        initial_forget_rates=forgets,
        initial_repetition_rates=repetitions,
        delta_coeffs=np.array([3, 20]),
        n_item=n_items,
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

    with ProgressBarCallback(env, check_freq) as callback:
        m.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.show()
    return m


def run_continuous_teaching(reward_type, gamma=1):
    global n_items, forgets, repetitions

    if LOAD_RATES:
        forget_rates = forgets
        repetition_rates = repetitions
    else:
        forget_rates, repetition_rates = produce_rates()

    env = ContinuousTeaching(
        t_max=100,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        n_item=n_items,
        tau=0.9,
        delta_coeffs=np.array([3, 20]),
        penalty_coeff=0.2,
        reward_type=reward_type,
        gamma=gamma
    )

    m = A2C(env)

    # iterations = env.t_max * 5e5
    iterations = 10e6
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        m.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.savefig('{}.png'.format(gamma))
    plt.show()
    return m


def curriculum_learning(reward_type, gamma, session_lengths=(50, 100)):
    global forgets, repetitions, LOAD_RATES

    if LOAD_RATES:
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
        iterations = 10e6 / len(session_lengths)
        check_freq = env.t_max
        for i in range(len(session_lengths)):
            env.t_max = session_lengths[i]
            if i > 0:
                gamma *= session_lengths[i] / session_lengths[i - 1]
            env.gamma = gamma
            with ProgressBarCallback(env, check_freq) as callback:
                m.learn(iterations, callback=callback)

        plt.plot([np.mean(r) for r in callback.hist_rewards])
        plt.savefig('curriculum_plots/{}_{}_{}.png'.format(EXPERIMENT_NAME, COMMIT_NAME, gamma))
        plt.clf()

    else:
        # TODO produce rates
        pass
    return m


if __name__ == "__main__":
    for i in [2, 3, 4, 5, 8]:
        print('Running on {}...'.format(i))
        if LOAD_RATES:
            forgets = pd.read_csv('data/forget_2', delimiter=',', header=None)
            repetitions = pd.read_csv('data/repetition_2', delimiter=',', header=None)
            forgets = np.array(forgets)[0]
            forgets = np.reshape(forgets, newshape=(n_users, n_items))
            forgets = forgets[:, :n_items // 2]
            repetitions = np.array(repetitions)[0]
            repetitions = np.reshape(repetitions, newshape=(n_users, n_items))
            repetitions = repetitions[:, :n_items // 2]

        model = curriculum_learning(types['exam_based'], i, session_lengths=(20, 50, 100))
        model.save('curriculum_runs/eb21_{}_{}'.format(COMMIT_NAME, i))
