import logging
import sys
import random

import numpy as np
import optuna

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback

from environments.continuous_teaching import ContinuousTeaching

from human_agents import generate_agents

n_users = 30
n_items = 30
random.seed(123)
test_users = random.sample(range(0, n_users), 3)
forget_rates, repetition_rates = generate_agents(n_users, n_items)


def run_on_test_users(env, policy):
    global test_users
    rewards = []
    actions = []
    env.penalty_coeff=0.0
    for user in test_users:
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


def suggest_params(trial):
    """ Learning hyper-parameters we want to optimise"""
    n_delta_coeffs = trial.suggest_int('n_coeffs', 1, 4)
    delta_coeffs = []
    last_coeff = 0
    for i in range(n_delta_coeffs):
        delta_coeff = trial.suggest_float('delta_coeff_{}'.format(i), last_coeff, 100)
        last_coeff = delta_coeff
        delta_coeffs += [delta_coeff]

    penalty_coeff = trial.suggest_float('penalty_coeff', 0.0, 1.0)

    has_shared_net = trial.suggest_categorical('has_shared_net', [True, False])
    shared_layers_dim = []
    if has_shared_net:
        n_shared_layers = trial.suggest_int('n_shared_layers', 1, 3)
        for i in range(n_shared_layers):
            shared_layer_dim = trial.suggest_int(
                'shared_layer_dim{}'.format(i),
                4,
                64
            )
            shared_layers_dim += [shared_layer_dim]
    n_layers = trial.suggest_int('n_layers', 1, 4)
    layers_dim = []
    for i in range(n_layers):
        layer_dim = trial.suggest_int('layer_dim{}'.format(i), 4, 128)
        layers_dim += [layer_dim]

    return {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-7, 1.),
        'constant_lr': trial.suggest_categorical('constant_lr', [True, False]),
        'normalize_advantage': trial.suggest_categorical('normalize_advantage', [True, False]),
        "net_arch": shared_layers_dim + [{'pi': layers_dim, 'vf': layers_dim}]
    }, {
        'n_coeffs': n_delta_coeffs,
        'delta_coeffs': np.array(delta_coeffs),
        'penalty_coeff': penalty_coeff,
    },


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    global forget_rates, repetition_rates
    model_params, env_params = suggest_params(trial)

    env = ContinuousTeaching(
        t_max=100,
        tau=0.9,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        **env_params
    )

    model = A2C(
        env,
        **model_params
    )

    iterations = env.t_max * 1000
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        model.learn(iterations, callback=callback)
    rewards, actions = run_on_test_users(env, model)
    return np.sum(rewards)


if __name__ == '__main__':

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "all_params_study"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
    )
    try:
        study.optimize(optimize_agent, n_trials=500)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
