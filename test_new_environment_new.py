import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from a2c.a2c import A2C
from a2c.old_a2c import A2C as OldA2C
from a2c.callback.teacher_exam import TeacherExamCallback
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
                break

    final_n_learned = rewards[-1] * env.n_item
    n_view = len(np.unique(np.asarray(actions)))
    print(f"{policy.__class__.__name__.lower()} | "
          f"final reward {int(final_n_learned)} | "
          f"precision {final_n_learned / n_view:.2f}")
    return actions, rewards


def main():

    # n_iter_per_session=30 |  with iterations=int(1e6); n_item=30
    #            => works (final reward=8 | Leitner=6 | Myopic=8)
    # n_iter_per_session=40 |  with iterations=int(1e6); n_item=30
    #            => works (final reward=10 | Leitner=8 | Myopic=8)
    # n_iter_per_session=50 |  with iterations=int(1e6); n_item=30
    #            => failed (final reward=5 | Leitner=8 | Myopic=9)
    # n_iter_per_session=50 |  with iterations=int(2e6); n_item=30
    #            => works (final reward=9 | Leitner=8 | Myopic=9)

    # For A2C only
    iterations = int(1e6)
    seed_a2c = 123

    # For everyone
    env = TeachingExam(
        init_forget_rate=0.02,
        rep_effect=0.2,
        n_item=20,
        learned_threshold=0.9,
        n_session=1,
        n_iter_per_session=30,
        time_per_iter=1,
        break_length=1)

    # Run Leitner ---------------------------------

    env.reset()

    policy = Leitner(env=env, delay_min=1, delay_factor=2)

    run(env=env, policy=policy)

    # Run Threshold ---------------------------------

    env.reset()

    policy = Threshold(env=env)

    run(env=env, policy=policy)

    # Run A2C ---------------------------------------
    env.reset()

    torch.manual_seed(seed_a2c)

    buffer_size = env.n_session*env.n_iter_per_session
    policy = A2C(env=env, buffer_size=buffer_size)
    # policy = OldA2C(env=env, n_steps=buffer_size)

    with TeacherExamCallback() as callback:
        policy.learn(iterations, callback=callback)

    run(env=env, policy=policy)


if __name__ == "__main__":
    main()
