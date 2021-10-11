import numpy as np
from matplotlib import pyplot as plt

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback
from environments.continuous_teaching import ContinuousTeaching
from human_agents import generate_agents


def run_one_episode(env, policy):
    rewards = []
    actions = []
    env.penalty_coeff=0.0
    obs = env.reset_keeping_user()
    while True:
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        actions.append(action)

        if done:
            obs = env.reset()
            break

    return rewards, actions


def teach_in_sessions(env, session_lengths):
    rewards = []
    actions = []
    models = []
    for session_length in session_lengths:
        model = A2C(env, seed=123)
        models += [model]
        env.t_max = session_length

        iterations = env.t_max * 1000
        check_freq = env.t_max

        with ProgressBarCallback(env, check_freq) as callback:
            model.learn(iterations, callback=callback)

        plt.plot([np.mean(r) for r in callback.hist_rewards])
        plt.show()

        r, a = run_one_episode(env, model)
        rewards += [np.array(r)]
        actions += [np.array(a)]

    return rewards, actions, models


if __name__ == '__main__':
    n_users = 10
    n_items = 140
    forget_rates, repetition_rates = generate_agents(n_users, n_items)
    env = ContinuousTeaching(
        t_max=100,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        n_item=n_items,
        tau=0.9,
        delta_coeffs=np.array([3, 20]),
        n_coeffs=2,
        penalty_coeff=0.4
    )
    teach_in_sessions(env, [10, 20, 40, 70, 100, 200])
