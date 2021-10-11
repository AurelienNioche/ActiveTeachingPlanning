import random

import matplotlib.pyplot as plt
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
n_items = 30


def produce_rates():
    global n_items, n_users
    forget_rates, repetition_rates = generate_agents(n_users, n_items, 123)
    print("forget", forget_rates.mean())
    print("repeat", repetition_rates.mean())
    return forget_rates, repetition_rates


def test_save_and_load():

    model = A2C(env=ContinuousTeaching())
    path = "bkp/a2c_test.p"
    model.save(path)
    model.load(path)


def test_continuous_teaching():
    global n_items

    forget_rates, repetition_rates = produce_rates()
    env = ContinuousTeaching(
        t_max=50,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        n_item=n_items,
        tau=0.9,
        delta_coeffs=np.array([3, 20]),
        n_coeffs=2,
        penalty_coeff=0.2,
        reward_coeff=20.,
        reward_type=types['exam_based']
    )

    model = A2C(env, seed=123)

    iterations = env.t_max * 20000
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        model.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.show()


def test_discontinuous_teaching():
    global n_items

    forget_rates, repetition_rates = produce_rates()
    env = DiscontinuousTeaching(
        tau=0.9,
        break_length=24 * 60 ** 2,
        time_per_iter=3,
        n_iter_per_session=100,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        delta_coeffs=np.array([3, 20]),
        penalty_coeff=0.3
    )
    model = A2C(env, seed=123)

    env_t_max = env.n_session * env.n_iter_per_session
    iterations = env_t_max * 10000
    check_freq = env_t_max

    with ProgressBarCallback(env, check_freq) as callback:
        model.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.show()


if __name__ == "__main__":

    test_continuous_teaching()
