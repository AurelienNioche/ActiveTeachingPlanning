import numpy as np


class Threshold:
    def __init__(self, env):
        self.env = env

    def act(self, obs):

        # If environment is ContinuousTeaching
        view = self.env.item_state[:, 1] > 0
        delta = self.env.item_state[view, 0]     # only consider already seen items
        rep = self.env.item_state[view, 1] - 1.  # only consider already seen items

        # print(delta)

        forget_rate = self.env.init_forget_rate[view] * \
            (1 - self.env.rep_effect[view]) ** rep

        log_p_recall = - forget_rate*delta

        # print(" ".join([f"{p:.3f}" for p in np.exp(log_p_recall)]))

        under_thr = log_p_recall <= self.env.log_thr

        if np.count_nonzero(under_thr) > 0:
            selection, = np.nonzero(view)
            action = selection[np.argmin(log_p_recall)]
        else:
            n_seen = np.count_nonzero(view)
            if n_seen == self.env.n_item:
                action = np.argmin(log_p_recall)
            else:
                action = n_seen
        # print("review:" if view[action] == 1 else "new:", action)
        # print("--")
        return action
