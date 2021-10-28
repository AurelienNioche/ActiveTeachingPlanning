from functools import partial
import torch
from torch import nn
import gym
from gym import spaces
import numpy as np


class FlattenExtractor(nn.Module):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super().__init__()

        self._observation_space = observation_space
        self.features_dim = spaces.utils.flatdim(observation_space)
        assert self.features_dim > 0

        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)


def init_weights(module: nn.Module, gain: float = 1) -> None:
    """
    Orthogonal initialization (used in PPO and A2C)
    """
    print("called ta mere", type(module))
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    else:
        print("skip")


def main():
    obs_space = gym.spaces.Box(low=-3, high=3, shape=(4, ))
    gain = np.sqrt(2)
    module = FlattenExtractor(obs_space)

    module.apply(partial(init_weights, gain=gain))
    #print(module.flatten.start_dim)
    #print(module.flatten.end_dim)
    #print("fils de pute")
    #obs = torch.ones((1, 4))
    #   print(module(obs.float()))



if __name__ == "__main__":
    main()