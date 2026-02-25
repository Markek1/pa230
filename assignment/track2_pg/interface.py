import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TEAM_NAME = "M"


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(ActorCriticNetwork, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_size, output_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        features = self.base(x)
        action_logits = self.actor(features)
        state_value = self.critic(features)

        return action_logits, state_value

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base(x)
        action_logits = self.actor(features)
        return torch.argmax(action_logits, dim=-1)

    def get_action(self, x: torch.Tensor):
        features = self.base(x)
        logits = self.actor(features)
        value = self.critic(features)

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value


class Policy:
    def __init__(self):
        self.model = None

    def load(self, model_path: str) -> None:
        obs_dim = 8
        action_dim = 4

        self.model = ActorCriticNetwork(obs_dim, action_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def act(self, observation: np.ndarray) -> int:
        if self.model is None:
            raise RuntimeError("Policy not loaded.")

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        return self.model.act(obs_tensor).item()


class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
