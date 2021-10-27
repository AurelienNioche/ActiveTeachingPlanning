from typing import Tuple

import torch
from torch import nn
from torch.distributions import Categorical


class CategoricalDistribution:
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: torch.Tensor) \
            -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()


class DiagGaussianDistribution:
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.
    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)
        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        # TODO: allow action dependent std
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init,
                               requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)
        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = torch.distributions.Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.
        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return self.sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        return self.sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.
        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    @staticmethod
    def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
        """
        Continuous actions are usually considered to be independent,
        so we can sum components of the ``log_prob`` or the entropy.
        :param tensor: shape: (n_batch, n_actions) or (n_batch,)
        :return: shape: (n_batch,)
        """
        if len(tensor.shape) > 1:
            tensor = tensor.sum(dim=1)
        else:
            tensor = tensor.sum()
        return tensor
