import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from a2c.a2c import A2C
from a2c.callback import ProgressBarCallback
from environments.teaching_exam import TeachingExam
from baseline_policies.leitner import Leitner
from baseline_policies.threshold import Threshold


def run(env, policy, seed=123):

    np.random.seed(seed)

    rewards = []
    actions = []

    obs = env.reset()

    with tqdm(total=env.n_iter_per_session * env.n_session) as pb:
        while True:
            action = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            actions.append(action)
            pb.update()
            if done:
                # # Simulate exam
                # obs, reward, done, _ = env.step(None)
                # rewards.append(reward)

                break

    final_n_learned = rewards[-1] * env.n_item
    n_view = len(np.unique(np.asarray(actions)))
    print(f"{policy.__class__.__name__.lower()} | "
          f"final reward {int(final_n_learned)} | "
          f"precision {final_n_learned / n_view:.2f}")
    return actions, rewards


def main():

    env = TeachingExam(
        init_forget_rate=0.02,
        rep_effect=0.2,
        n_item=30,
        learned_threshold=0.9,
        n_session=1,
        n_iter_per_session=100,
        time_per_iter=1,
        break_length=1)

    policy = Leitner(env=env, delay_min=1, delay_factor=2)

    run(env=env, policy=policy)

    env.reset()

    policy = Threshold(env=env)

    run(env=env, policy=policy)

    env.reset()

    policy = A2C(env)

    # iterations = env.t_max * 5e5
    iterations = int(10e6)
    check_freq = env.total_iteration

    with ProgressBarCallback(env, check_freq) as callback:
        policy.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.show()

    run(env=env, policy=policy)

    # m = A2C(env)
    #
    # # iterations = env.t_max * 5e5
    # iterations = int(10e6)
    # check_freq = env.total_iteration
    #
    # with ProgressBarCallback(env, check_freq) as callback:
    #     m.learn(iterations, callback=callback)
    #
    # plt.plot([np.mean(r) for r in callback.hist_rewards])
    # plt.show()

    # teacher = A2C(env=env)


if __name__ == "__main__":
    main()
