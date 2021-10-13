import pandas as pd
import scipy.special
import numpy as np


def generate_learners_parameterization(n_users, n_items, seed):
    np.random.seed(seed)
    df_param = pd.read_csv("data/param_exp_data.csv", index_col=0)
    mu = np.array([df_param.loc["unconstrained", f"mu{i}"]
                   for i in (1, 2)])
    sig_users = np.array([df_param.loc["unconstrained", f"sigma_u{i}"]
                          for i in (1, 2)])
    sig_items = np.array([df_param.loc["unconstrained", f"sigma_w{i}"]
                          for i in (1, 2)])

    z_user = np.random.normal(np.zeros(2), sig_users, size=(n_users, 2))
    z_item = np.random.normal(np.zeros(2), sig_items, size=(n_items, 2))
    initial_forget_rates = np.zeros((n_users, n_items))
    repetition_effect_rates = np.zeros((n_users, n_items))
    for i in range(n_users):
        initial_forget_rates[i] = mu[0] + z_user[i, 0] + z_item[:, 0]
        repetition_effect_rates[i] = mu[1] + z_user[i, 1] + z_item[:, 1]

    initial_forget_rates = np.exp(initial_forget_rates)
    repetition_effect_rates = scipy.special.expit(repetition_effect_rates)

    return initial_forget_rates, repetition_effect_rates
