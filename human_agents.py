import pandas as pd
import scipy.special
import numpy as np


def generate_agents(n_users, n_items, seed=None):
    if seed is not None:
        np.random.seed(seed)
    df_param = pd.read_csv("data/param_exp_data.csv", index_col=0)
    mu = np.array([df_param.loc["unconstrained", f"mu{i}"] for i in range(1, 3)])
    sig_users = np.array([df_param.loc["unconstrained", f"sigma_u{i}"] for i in range(1, 3)])
    sig_items = np.array([df_param.loc["unconstrained", f"sigma_w{i}"] for i in range(1, 3)])

    z_user = np.random.normal(np.zeros(2), sig_users, size=(n_users, 2))
    z_item = np.random.normal(np.zeros(2), sig_items, size=(n_items, 2))
    initial_forget_rates, initial_repetition_rates = [], []
    for i in range(n_users):
        initial_forget_rates += [mu[0] + z_user[i, 0] + z_item[:, 0]]
    for i in range(n_users):
        initial_repetition_rates += [mu[1] + z_user[i, 1] + z_item[:, 1]]

    initial_forget_rates = np.exp(np.array(initial_forget_rates))

    initial_repetition_rates = scipy.special.expit(np.array(initial_repetition_rates))

    return initial_forget_rates, initial_repetition_rates

