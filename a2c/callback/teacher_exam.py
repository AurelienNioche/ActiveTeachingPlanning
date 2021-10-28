import copy
from tqdm import tqdm


class TeacherExamCallback:

    def __init__(self):

        self.pbar = None
        self.model = None

    def init_callback(self, model) -> None:
        """
        Initialize the callback by saving references to the
        RL model
        """
        self.model = model

    def on_training_start(self, total_timesteps) -> None:
        self.pbar.reset(total=total_timesteps)

    def on_training_end(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        reward = int(self.model.env.reward*self.model.env.n_item)
        self.pbar.set_postfix({"reward": f"{reward}"})
        pass

    def on_step(self):
        """
        This method will be called by the model after each call to ``env.step()``.
        :return: If the callback returns False, training is aborted early.
        """
        # Update the progress bar:
        self.pbar.n = self.model.num_timesteps
        self.pbar.update(0)
        return True

    def __enter__(self):
        # create the progress bar
        self.pbar = tqdm()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.n = self.model.num_timesteps
        self.pbar.update(0)
        self.pbar.close()
