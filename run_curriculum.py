import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback

from environments.continuous_teaching import ContinuousTeaching
from environments import reward_types
from generate_learners.generate_learners import \
    generate_learners_parameterization

from config.run_curriculum import *

sns.set()

os.makedirs(BKP_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)


def curriculum_learning(reward_type, gamma, session_lengths=(50, 100)):

    forgets, repetitions = generate_learners_parameterization(
        n_users=N_USERS, n_items=N_ITEMS, seed=SEED_PARAM_LEARNERS)

    env = ContinuousTeaching(
        t_max=100,
        initial_forget_rates=forgets,
        initial_repetition_rates=repetitions,
        n_item=N_ITEMS,
        tau=0.9,
        delta_coeffs=np.array([3, 20]),
        penalty_coeff=0.2,
        reward_type=reward_type,
        gamma=gamma
    )
    m = A2C(env)
    iterations = int(10e6 / len(session_lengths))
    check_freq = env.t_max
    for i in range(len(session_lengths)):
        env.t_max = session_lengths[i]
        if i > 0:
            gamma *= session_lengths[i] / session_lengths[i - 1]
        env.gamma = gamma
        with ProgressBarCallback(env, check_freq) as callback:
            m.learn(iterations, callback=callback)

    means = np.array([np.mean(r) for r in callback.hist_rewards])
    plt.plot(means)
    np.savetxt(f'{BKP_FOLDER}/{EXPERIMENT_NAME}_{COMMIT_NAME}_{str(session_lengths)}_{gamma}.csv', means, delimiter=',')
    plt.savefig(f'{FIG_FOLDER}/{EXPERIMENT_NAME}_{COMMIT_NAME}_{str(session_lengths)}_{gamma}.png')
    plt.clf()

    return m


def main():
    for session_lengths in ALL_SESSION_LENGTHS:
        for gamma in [2, 3, 4, 5, 8]:
            print(session_lengths)
            print(f'Running with gamma={gamma}...')
            model = curriculum_learning(
                reward_type=reward_types.EXAM_BASED,
                gamma=gamma,
                session_lengths=session_lengths)
            model.save(f'{BKP_FOLDER}/{EXPERIMENT_NAME}_{COMMIT_NAME}_{str(session_lengths)}_{gamma}.p')


if __name__ == "__main__":
    main()
