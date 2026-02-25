import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

TEAM_NAME = "M"


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> torch.Tensor:
        q_values = self.forward(x)
        return torch.argmax(q_values, dim=-1)


class Policy:
    def __init__(self):
        self.model = None

    def load(self, model_path: str) -> None:
        obs_dim = 8
        action_dim = 4

        self.model = QNetwork(obs_dim, action_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def act(self, observation: np.ndarray) -> int:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)

        action_idx = self.model.act(obs_tensor)
        return action_idx.item()


class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
