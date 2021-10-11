import logging
import sys
import random

import numpy as np
import optuna

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback

from environments.continuous_teaching import ContinuousTeaching

from human_agents import generate_agents

n_users = 10
n_items = 140
random.seed(123)
test_users = random.sample(range(0, n_users), 3)
forget_rates, repetition_rates = generate_agents(n_users, n_items)


def run_on_test_users(env, policy):
    global test_users
    rewards = []
    actions = []
    env.reward_coeff = 1
    for user in test_users:
        env.penalty_coeff=0.0
        obs = env.reset_for_new_user(user)
        while True:
            action = policy.act(obs)
            obs, reward, done, info = env.step(action)

            rewards.append(reward)
            actions.append(action)

            if done:
                obs = env.reset()
                break

    return rewards, actions


def optimize_agent(trial, ):
    """ Train the model and optimize"""

    global forget_rates, repetition_rates
    coeff = trial.suggest_float('coeff', 0, n_items)

    env = ContinuousTeaching(
        t_max=100,
        tau=0.9,
        n_item=n_items,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        delta_coeffs=np.array([3, 20]),
        n_coeffs=2,
        reward_coeff=coeff
    )
    model = A2C(
        env,
        learning_rate=5e-4,
        constant_lr=True,
        normalize_advantage=False,
    )

    iterations = env.t_max * 2000
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        model.learn(iterations, callback=callback)
    rewards, actions = run_on_test_users(env, model)
    return np.sum(rewards)


if __name__ == '__main__':

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "reward_coefficient_study"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
    )
    try:
        study.optimize(optimize_agent, n_trials=200)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
