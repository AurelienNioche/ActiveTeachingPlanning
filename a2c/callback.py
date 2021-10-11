import copy

from tqdm.autonotebook import tqdm
import numpy as np


class ProgressBarCallback:
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, env, check_freq):

        self.env = copy.deepcopy(env)
        self.check_freq = check_freq

        self.hist_rewards = []

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
        pass

    def on_step(self):
        """
        This method will be called by the model after each call to ``env.step()``.
        :return: If the callback returns False, training is aborted early.
        """

        # Update the progress bar:
        self.pbar.n = self.model.num_timesteps
        self.pbar.update(0)

        # Evaluate
        if self.model.num_timesteps % self.check_freq == 0:
            obs = self.env.reset()
            rewards = []
            while True:
                action = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)

                rewards.append(reward)

                if done:
                    break
            self.hist_rewards.append(rewards)
            self.pbar.set_postfix({"average-reward": f"{np.mean(rewards):.4f}",
                                   "max-reward": f"{np.max(rewards):.4f}",
                                   "min-reward": f"{np.min(rewards):.4f}"})
        return True

    def __enter__(self):
        # create the progress bar
        self.pbar = tqdm()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.n = self.model.num_timesteps
        self.pbar.update(0)
        self.pbar.close()
