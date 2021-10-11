import pandas as pd
import numpy as np
import torch


def get_experimental_data():

    df = pd.read_csv("data/data.csv")

    # Copy actual item ID in a new column
    df["item"] = pd.factorize(df.character)[0]

    # Total number of user
    n_u = len(df.user.unique())

    # Number of observations per user
    n_o_by_u = np.zeros(shape=n_u, dtype=int)
    for u, (user, user_df) in enumerate(df.groupby("user")):
        # Do not count first presentation
        n_o_by_u[u] = len(user_df) - len(user_df.item.unique())

        # Total number of observation
    n_obs = n_o_by_u.sum()

    # Prepare data containers

    # Replies (1: success, 0: error)
    y = np.zeros(shape=n_obs, dtype=int)
    # Time elapsed since the last presentation of the same item (in seconds)
    x = np.zeros(shape=n_obs, dtype=float)
    # Number of repetition (number of presentation - 1)
    r = np.zeros(shape=n_obs, dtype=int)
    # Item ID
    w = np.zeros(shape=n_obs, dtype=int)
    # User ID
    u = np.zeros(shape=n_obs, dtype=int)

    # Fill the containers `y`, `x`, `r`, `w`, `u`
    idx = 0
    for i_u, (user, user_df) in enumerate(df.groupby("user")):

        # Extract data from user `u`
        user_df = user_df.sort_values(by="ts_reply")
        seen = user_df.item.unique()
        w_u = user_df.item.values
        ts_u = user_df.ts_reply.values
        y_u = user_df.success.values

        # Initialize counts of repetition for each words at -1
        counts = {word: -1 for word in seen}
        # Initialize time of last presentation at None
        last_pres = {word: None for word in seen}

        # Number of observations for user `u` including first presentations
        n_obs_u_incl_first = len(user_df)

        # Number of repetitions for user `u`
        r_u = np.zeros(n_obs_u_incl_first)
        # Time elapsed since last repetition for user `u`
        x_u = np.zeros(n_obs_u_incl_first)

        # Loop over each entry for user `u`:
        for i in range(n_obs_u_incl_first):

            # Get info for iteration `i`
            word = w_u[i]
            ts = ts_u[i]
            r_u[i] = counts[word]

            # Compute time elasped since last presentation
            if last_pres[word] is not None:
                x_u[i] = ts - last_pres[word]

            # Update count of repetition
            counts[word] += 1
            # Update last presentation
            last_pres[word] = ts

        # Keep only observations that are not the first presentation of an item
        to_keep = r_u >= 0
        y_u = y_u[to_keep]
        r_u = r_u[to_keep]
        w_u = w_u[to_keep]
        x_u = x_u[to_keep]

        # Number of observations for user `u` excluding first presentations
        n_obs_u = len(y_u)

        # Fill containers
        y[idx:idx + n_obs_u] = y_u
        x[idx:idx + n_obs_u] = x_u
        w[idx:idx + n_obs_u] = w_u
        r[idx:idx + n_obs_u] = r_u
        u[idx:idx + n_obs_u] = i_u

        # Update index
        idx += n_obs_u

    data = {
        'u': u, 'w': w,
        'x': torch.from_numpy(x.reshape(-1, 1)),
        'r': torch.from_numpy(r.astype(float).reshape(-1, 1)),
        'y': torch.from_numpy(y.astype(float).reshape(-1, 1))
    }

    n_w = len(np.unique(w))
    n_o_max = n_o_by_u.max()
    n_o_min = n_o_by_u.min()
    print("number of user", n_u)
    print("number of items", n_w)
    print("total number of observations (excluding first presentation)", n_obs)
    print("minimum number of observation for a single user", n_o_min)
    print("maximum number of observation for a single user", n_o_max)

    return data
