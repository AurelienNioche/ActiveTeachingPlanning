from a2c.a2c import A2C
from environments.supervisor_env import SupervisorEnv

#
# class SupervisorCallback:
#
#     def __init__(self):
#
#         self.model = None
#
#     def init_callback(self, model) -> None:
#         """
#         Initialize the callback by saving references to the
#         RL model
#         """
#         self.model = model
#
#     def on_training_start(self, total_timesteps) -> None:
#         pass
#
#     def on_training_end(self) -> None:
#         pass
#
#     def on_rollout_start(self) -> None:
#         pass
#
#     def on_rollout_end(self) -> None:
#         pass
#
#     def on_step(self):
#         """
#         This method will be called by the model after each call to ``env.step()``.
#         :return: If the callback returns False, training is aborted early.
#         """
#
#         self.model.env.
#         return True


def teach_the_teacher():

    supervisor = A2C(env=SupervisorEnv())
    supervisor.learn(total_timesteps=10000)


def main():

    teach_the_teacher()


if __name__ == "__main__":
    main()
