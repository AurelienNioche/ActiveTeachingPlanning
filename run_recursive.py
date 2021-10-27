from a2c.a2c import A2C
from environments.supervisor_env import SupervisorEnv
from a2c.callback.supervisor import SupervisorCallback


def teach_the_teacher():
    supervisor = A2C(env=SupervisorEnv(teaching_iterations=int(1e4)), n_steps=1)
    with SupervisorCallback(freq_save=500) as callback:
        supervisor.learn(total_timesteps=int(1e4),  callback=callback)


def main():

    teach_the_teacher()


if __name__ == "__main__":
    main()
