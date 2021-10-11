import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback

from environments.continuous_teaching import ContinuousTeaching

sns.set()


def main():

    env = ContinuousTeaching(t_max=100, alpha=0.2, tau=0.9)
    model = A2C(env, seed=123)

    iterations = env.t_max * 1000
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        model.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.show()


if __name__ == "__main__":
    main()
