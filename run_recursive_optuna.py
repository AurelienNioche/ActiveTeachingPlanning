import os
import numpy as np
import optuna
import functools
import json

from a2c.a2c import A2C
from environments.teaching_exam import TeachingExam as TeacherEnv


def run(env, policy, seed=123):

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


def objective(trial, teacher, env_kwargs, teaching_iterations,
              teacher_bkp_file):

    n_iter_per_session = trial.suggest_int('n_iter_per_session', 1, 100)
    teacher.env = TeacherEnv(n_iter_per_session=n_iter_per_session,
                             **env_kwargs)
    teacher.buffer_size = teacher.env.n_iter_per_session * teacher.env.n_session

    teacher.learn(teaching_iterations, callback=None)

    teacher.save(teacher_bkp_file)

    return run(teacher.env, teacher)


def teach_the_teacher(epochs=500, force=False):

    teaching_iterations = int(1e4)

    bkp_folder = "bkp/run_recursive_optuna"
    optuna_bkp_file = f"{bkp_folder}/optuna.db"
    teacher_bkp_file = f"{bkp_folder}/teacher.p"
    env_kwargs_bkp_file = f"{bkp_folder}/env_kwargs.json"

    os.makedirs(bkp_folder, exist_ok=True)

    if os.path.exists(teacher_bkp_file) and not force:
        teacher = A2C.load(teacher_bkp_file)
        with open(env_kwargs_bkp_file, 'r') as f:
            env_kwargs = json.load(f)
        # study = optuna.load_study(storage=f"sqlite:///{optuna_bkp_file}",
        #                           study_name="optuna_teach_the_teacher")
    else:
        if os.path.exists(optuna_bkp_file) and force:
            os.remove(optuna_bkp_file)

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

        study = optuna.create_study(direction='maximize',
                                    storage=f"sqlite:///{optuna_bkp_file}",
                                    study_name="optuna_teach_the_teacher")

        study.optimize(functools.partial(objective,
                                         teacher=teacher,
                                         teaching_iterations=teaching_iterations,
                                         env_kwargs=env_kwargs,
                                         teacher_bkp_file=teacher_bkp_file),
                       n_trials=epochs)

    env = TeacherEnv(n_iter_per_session=100, **env_kwargs)
    print("reward", run(policy=teacher, env=env))


if __name__ == "__main__":
    teach_the_teacher()
