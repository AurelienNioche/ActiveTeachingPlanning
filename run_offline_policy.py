import os
import numpy as np
import functools
import json

from a2c.a2c_offline import A2C
from environments.teaching_exam import TeachingExam as TeacherEnv
from baseline_policies.threshold import Threshold


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


def teach_the_teacher(epochs=500, force=True):

    teaching_iterations = int(1e4)

    bkp_folder = "bkp/run_offline_policy"

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

        env = TeacherEnv(n_iter_per_session=100, **env_kwargs)
        teacher = A2C(env=env)

        rewards = []

        for i in range(epochs):

            # teacher.env = TeacherEnv(n_iter_per_session=100,
            #                          **env_kwargs)
            # teacher.buffer_size = teacher.env.n_iter_per_session * teacher.env.n_session

            if i % 2 == 0:
                rollout_policy = Threshold(env=teacher.env)
                teacher.learn(teaching_iterations, callback=None,
                              rollout_policy=rollout_policy)
            else:
                teacher.learn(teaching_iterations, callback=None,
                              rollout_policy=None)
            print(teacher.env.reward * teacher.env.n_item)

            teacher.save(teacher_bkp_file)
            reward = evaluate(teacher.env, teacher)

            # reward = _train(n_iter_per_session=100)
            print(f"trial {i} | reward={reward}")

            # actions.append(n_iter_per_session)
            rewards.append(reward)

        np.save(file=f'{bkp_folder}/rewards.npy', arr=np.asarray(rewards))
        # np.save(file=f'{bkp_folder}/actions.npy', arr=np.asarray(actions))

    env = TeacherEnv(n_iter_per_session=100, **env_kwargs)
    print("Reward", evaluate(policy=teacher, env=env))


if __name__ == "__main__":
    teach_the_teacher()
