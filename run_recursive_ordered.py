import os
import numpy as np
import functools
import json

from a2c.a2c import A2C
from environments.teaching_exam import TeachingExam as TeacherEnv


def generate_progression(n_trials, low, high, discrete=False):
    if discrete:
        high = high-1
    a = np.arange(n_trials)
    a = low + (high - low) * (a - a.min()) / (a.max() - a.min())

    if discrete:
        a = np.round(a).astype(int)
    return a


def evaluate(env, policy, seed=123):

    np.random.seed(seed)

    rewards = []
    actions = []

    obs = env.reset()

    while True:
        action = policy.act(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break

    final_n_learned = rewards[-1] * env.n_item
    # n_view = len(np.unique(np.asarray(actions)))
    # print(f"{policy.__class__.__name__.lower()} | "
    #       f"final reward {int(final_n_learned)} | "
    #       f"precision {final_n_learned / n_view:.2f}")
    return final_n_learned


def train(n_iter_per_session, teacher, env_kwargs, teaching_iterations,
          teacher_bkp_file):

    teacher.env = TeacherEnv(n_iter_per_session=n_iter_per_session,
                             **env_kwargs)
    teacher.buffer_size = teacher.env.n_iter_per_session * teacher.env.n_session

    teacher.learn(teaching_iterations, callback=None)

    teacher.save(teacher_bkp_file)
    return evaluate(teacher.env, teacher)


def teach_the_teacher(epochs=500, force=False):

    teaching_iterations = int(1e4)

    bkp_folder = "bkp/run_recursive_ordered"

    teacher_bkp_file = f"{bkp_folder}/teacher.p"
    env_kwargs_bkp_file = f"{bkp_folder}/env_kwargs.json"

    os.makedirs(bkp_folder, exist_ok=True)

    if os.path.exists(teacher_bkp_file) and not force:

        teacher = A2C.load(teacher_bkp_file)
        with open(env_kwargs_bkp_file, 'r') as f:
            env_kwargs = json.load(f)

    else:

        env_kwargs = dict(
            init_forget_rate=0.02,
            rep_effect=0.2,
            n_item=30,
            learned_threshold=0.9,
            n_session=1,
            time_per_iter=1,
            break_length=1
        )
        with open(env_kwargs_bkp_file, 'w') as f:
            json.dump(env_kwargs, f)

        env = TeacherEnv(n_iter_per_session=-1, **env_kwargs)
        teacher = A2C(env=env)

        _train = functools.partial(
            train,
            teacher=teacher,
            teaching_iterations=teaching_iterations,
            env_kwargs=env_kwargs,
            teacher_bkp_file=teacher_bkp_file)

        pgr = generate_progression(epochs, low=1, high=101, discrete=True)

        rewards = []
        actions = []

        for i in range(epochs):

            n_iter_per_session = pgr[i]

            reward = _train(n_iter_per_session=n_iter_per_session)
            print(f"trial {i} | n_iter_per_session={n_iter_per_session} | reward={reward}")

            actions.append(n_iter_per_session)
            rewards.append(reward)

        np.save(file=f'{bkp_folder}/rewards.npy', arr=np.asarray(rewards))
        np.save(file=f'{bkp_folder}/actions.npy', arr=np.asarray(actions))

    env = TeacherEnv(n_iter_per_session=100, **env_kwargs)
    print("Reward", evaluate(policy=teacher, env=env))


if __name__ == "__main__":
    teach_the_teacher()
