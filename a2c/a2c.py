import os
from typing import Any, Dict, Optional, Union, Tuple, List
from functools import partial

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import gym
from gym import spaces

from . nn import MlpExtractor
from . distribution import CategoricalDistribution, DiagGaussianDistribution

from . callback.supervisor import SupervisorCallback
from . callback.teacher import TeacherCallback
from . callback.progress_bar import SimpleProgressBarCallback

MaybeCallback = Union[SimpleProgressBarCallback, TeacherCallback, SupervisorCallback, None]


class A2C(nn.Module):
    """
    Advantage Actor Critic (A2C)
    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)
    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752
    :param env: The environment to learn from
    :param learning_rate: The learning rate
    :param constant_lr: Constant learning rate (yes or no)
    :param optimizer_name: The name of the optimizer to use,
        ``torch.optim.RMSprop`` ('RMSprop') by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    # :param normalize_advantage: Whether to normalize or not the advantage
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn_name: Name of the activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    # :param squash_output: For continuous actions, whether the output is squashed
    #     or not using a ``tanh()`` function.
    :param seed: Seed for the pseudo random generators
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 7e-4,
        constant_lr: bool = False,
        optimizer_name: str = "RMSprop",
        optimizer_kwargs: dict = None,
        buffer_size: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # normalize_advantage: bool = False,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn_name: str = "Tanh",
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        # squash_output: bool = False,
        seed: int = 123,
    ):

        super().__init__()

        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0

        self.seed = seed

        self.learning_rate = learning_rate

        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        # self._current_progress_remaining = 1

        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._last_obs = np.zeros((1, *self.env.observation_space.shape),
                                  dtype=np.float32)
        self._last_done = False

        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # self.normalize_advantage = normalize_advantage

        # Constant learning rate
        self.constant_lr = constant_lr

        # Matters for continuous action space only ('Box')
        self.log_std_init = log_std_init
        # self.squash_output = squash_output

        # Seed numpy RNG
        np.random.seed(seed)
        # seed torch RNG
        torch.manual_seed(seed)

        self.action_space.seed(seed)
        self.env.seed(seed)

        self.actions = None
        self.rewards = None
        self.dones = None
        self.observations = None

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]
        self.net_arch = net_arch                      # for saving
        self.activation_fn_name = activation_fn_name  # for saving
        activation_fn = getattr(torch.nn, activation_fn_name)
        self.ortho_init = ortho_init

        features_dim = spaces.utils.flatdim(self.observation_space)
        self.mlp_extractor = MlpExtractor(
            features_dim,
            net_arch=self.net_arch,
            activation_fn=activation_fn)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Action distribution
        if isinstance(self.action_space, spaces.Discrete):
            self.action_dist = CategoricalDistribution(self.action_space.n)
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi)
        elif isinstance(self.action_space, spaces.Box):
            assert len(self.action_space.shape) == 1, "Error: the action space must be a vector"
            self.action_dist = DiagGaussianDistribution(self.action_space.shape[0])
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                log_std_init=self.log_std_init)
        else:
            raise Exception("Action space not compatible with current implementation")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:  # true by defaut
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                # self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # `eps`: RMSProp epsilon:
        # it stabilizes square root computation in denominator
        # of RMSProp update
        optimizer_class = getattr(torch.optim, optimizer_name)
        if optimizer_name == 'RMSprop':
            default_optimizer_kwargs = dict(alpha=0.99, eps=1e-5, weight_decay=0)
        else:
            default_optimizer_kwargs = dict(eps=1e-5)
        if optimizer_kwargs is not None:
            default_optimizer_kwargs.update(optimizer_kwargs)
        optimizer_kwargs = default_optimizer_kwargs

        self.optimizer_name = optimizer_name      # for saving
        self.optimizer_kwargs = optimizer_kwargs  # for saving
        self.optimizer = optimizer_class(
            self.parameters(),
            lr=self.learning_rate, **self.optimizer_kwargs)

        self.buf_obs = np.zeros((1, *env.observation_space.shape),
                                dtype=np.float32)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def update_policy(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Maybe maintain constant the learning rate
        if self.constant_lr:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        indices = np.random.permutation(self.buffer_size)

        observations = torch.as_tensor(self.observations[indices])
        actions = torch.as_tensor(self.actions[indices])
        # Convert discrete action from float to long
        actions = actions.long().flatten()

        values, log_prob, entropy = self.evaluate_actions(
            observations, actions)
        values = values.flatten()

        # Compute advantages
        with torch.no_grad():

            # Compute value for the last timestep
            obs_tensor = torch.as_tensor(self._last_obs)
            _, last_value, _ = self.forward(obs_tensor)

            done = self._last_done
            last_value = last_value.numpy()

        rewards = self.rewards
        dones = self.dones
        detached_values = np.zeros_like(self.actions)
        advantages = np.zeros_like(self.actions)
        detached_values[:] = values.detach().clone().numpy().flatten()[np.argsort(indices)]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_values = detached_values[step + 1]
            delta = rewards[step] + self.gamma * next_values * next_non_terminal - detached_values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + detached_values

        tensor_advantages = torch.as_tensor(advantages[indices].flatten())
        tensor_returns = torch.as_tensor(returns[indices])

        # # Normalize advantage (not present in the original implementation)
        # if self.normalize_advantage:
        #     advantages = \
        #         (advantages - advantages.mean()) \
        #         / (advantages.std() + 1e-8)

        # Policy gradient loss
        policy_loss = -(tensor_advantages * log_prob).mean()

        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(tensor_returns, values)

        # if self.num_timesteps >= 188910:
        #     print(self.num_timesteps, "value loss", value_loss, "policy loss",
        #           policy_loss)
        #     print("returns",
        #           " ".join([f'{v:.3f}' for v in returns]))
        #     print("advantages",
        #           " ".join([f'{v:.3f}' for v in advantages]))
        #     print("log_prob",
        #           " ".join([f'{v:.3f}' for v in log_prob.detach().numpy()]))
        #     print("actions",
        #           " ".join([f'{v}' for v in self.actions]))

        # Entropy loss favor exploration
        entropy_loss = -torch.mean(entropy)

        loss = policy_loss \
            + self.ent_coef * entropy_loss \
            + self.vf_coef * value_loss

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Clip grad norm
        nn.utils.clip_grad_norm_(self.parameters(),
                                 self.max_grad_norm)
        self.optimizer.step()

    def collect_rollouts(self, callback: MaybeCallback) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :return: True if function returned with at least `n_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        obs_shape = self.observation_space.shape
        self.actions = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros_like(self.actions)
        self.dones = np.zeros_like(self.actions)
        self.observations = np.zeros((self.buffer_size,) + obs_shape, dtype=np.float32)

        if callback is not None:
            callback.on_rollout_start()

        for i in range(self.buffer_size):

            with torch.no_grad():
                # Convert to pytorch tensor
                obs_tensor = torch.as_tensor(self._last_obs)
                action, value, log_prob = self.forward(obs_tensor)
            action = action.numpy()

            clipped_action = action
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_action = np.clip(action, self.action_space.low, self.action_space.high)

            # Perform action
            new_obs, reward, done, _ = self.env.step(clipped_action)
            if done:
                new_obs = self.env.reset()

            self.buf_obs[0] = new_obs

            self.num_timesteps += 1

            if callback is not None and callback.on_step() is False:
                return False

            self.actions[i] = action
            self.rewards[i] = reward
            self.observations[i] = self._last_obs[0].copy()
            self.dones[i] = self._last_done

            self._last_obs = self.buf_obs.copy()
            self._last_done = done

        if callback is not None:
            callback.on_rollout_end()

        return True

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            reset_num_timesteps: bool = True):

        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps)

        if callback is not None:
            callback.on_training_start(total_timesteps)

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(callback)

            if continue_training is False:
                break

            # Compute current progress remaining (starts from 1 and ends to 0)
            # self._current_progress_remaining = \
            #     1.0 - self.num_timesteps / float(total_timesteps)

            self.update_policy()

        if callback is not None:
            callback.on_training_end()
        return self

    def _setup_learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            reset_num_timesteps: bool = True,
    ) -> Tuple[int, MaybeCallback]:
        """
        Initialize different variables needed for training.
        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :return:
        """

        if reset_num_timesteps:
            self.num_timesteps = 0

        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps:
            self._last_obs[0] = self.env.reset()
            self._last_done = False

        # Create eval callback if needed
        if callback is not None:
            callback.init_callback(self)

        return total_timesteps, callback

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def forward_action_only(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions

    def _get_latent(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        features = obs.float()
        latent_pi, latent_vf = self.mlp_extractor(features)
        return latent_pi, latent_vf

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Union[CategoricalDistribution, DiagGaussianDistribution]:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        else:
            raise ValueError("Action distribution not compatible with the current implementation")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        assert not isinstance(observation, dict), "using ObsDictWrapper but no supported here"
        observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = torch.as_tensor(observation)
        with torch.no_grad():
            latent_pi, _ = self._get_latent(observation)
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions = distribution.get_actions(deterministic=deterministic)

        # Convert to numpy
        actions = actions.numpy()

        # In case of continuous actions
        if isinstance(self.action_space, gym.spaces.Box):
            # if self.squash_output:
            #     # Rescale to proper domain when using squashing
            #     actions = self.unscale_action(actions)
            # else:
            # Actions could be on arbitrary scale, so clip the actions to avoid
            # out of bound error (e.g. if sampling from a Gaussian distribution)
            actions = np.clip(actions, self.action_space.low,
                              self.action_space.high)

        return actions

    def act(self, observation):
        """
        Alias for deterministic prediction
        """
        return self.predict(observation, deterministic=True)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _get_constructor_parameters(self) -> Dict[str, Any]:

        return dict(
            env=self.env,
            learning_rate=self.learning_rate,
            optimizer_name=self.optimizer_name,
            optimizer_kwargs=self.optimizer_kwargs,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            # normalize_advantage=self.normalize_advantage,
            ortho_init=self.ortho_init,
            log_std_init=self.log_std_init,
            net_arch=self.net_arch,
            activation_fn_name=self.activation_fn_name,
            seed=self.seed)

    def save(self, path: str) -> None:
        """
        Save model to a given location.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.state_dict(),
                    "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls, path: str):
        """
        Load model from path.
        """
        saved_variables = torch.load(path)
        # Create policy object
        model = cls(**saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        return model
