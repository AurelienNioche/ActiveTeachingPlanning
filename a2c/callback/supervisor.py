import os
from tqdm import tqdm


class SupervisorCallback:

    def __init__(self, freq_save=500, bkp_folder="bkp/supervisor"):

        self.pbar = None
        self.model = None

        self.freq_save = freq_save
        self.bkp_folder = bkp_folder
        os.makedirs(bkp_folder, exist_ok=True)

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
        # print([p for p in self.model.parameters() if p.requires_grad])
        self.pbar.set_postfix({
            "step": self.model.env.step_counter,
            "n_iter": self.model.env.n_iter,
            "reward": self.model.env.reward
        })
        iteration = self.model.env.step_counter
        if iteration % self.freq_save == 0:
            print("save")
            self.model.save(f"{self.bkp_folder}/supervisor_{iteration}.p")
            self.model.env.teacher.save(f"{self.bkp_folder}/teacher_{iteration}.p")
        return True

    def __enter__(self):
        # create the progress bar
        self.pbar = tqdm()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.n = self.model.num_timesteps
        self.pbar.update(0)
        self.pbar.close()
