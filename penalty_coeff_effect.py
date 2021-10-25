import logging
import sys
import random

import numpy as np
import optuna

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback

from environments.continuous_teaching import ContinuousTeaching

from generate_learners.generate_learners \
    import generate_learners_parameterization

N_USERS = 30
N_ITEMS = 30
SEED = 123

test_users = random.sample(range(0, N_USERS), 3)
forget_rates, repetition_rates = \
    generate_learners_parameterization(N_USERS, N_ITEMS)

random.seed(SEED)


def run_on_test_users(env, policy):

    rewards = []
    actions = []
    for user in test_users:
        env.penalty_coeff = 0.0
        obs = env.reset_for_new_user(user)
        while True:
            action = policy.act(obs)
            obs, reward, done, info = env.step(action)

            rewards.append(reward)
            actions.append(action)

            if done:
                env.reset()
                break

    return rewards, actions


def optimize_coeff(trial):
    """ Learning hyper-parameters we want to optimise"""
    coeff = trial.suggest_float('penalty_coeff', 0.0, 1.0)
    return {
        'penalty_coeff': coeff,
    }


def optimize_agent(trial, ):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """

    env_params = optimize_coeff(trial)

    env = ContinuousTeaching(
        t_max=100,
        tau=0.9,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        delta_coeffs=np.array([40, 7]),
        **env_params
    )

    model = A2C(
        env,
        learning_rate=5e-4,
        constant_lr=True,
        normalize_advantage=False,
    )

    iterations = env.t_max * 1000
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        model.learn(iterations, callback=callback)
    rewards, actions = run_on_test_users(env, model)
    return np.sum(rewards)


if __name__ == '__main__':

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "penalty_coeff_study"
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
