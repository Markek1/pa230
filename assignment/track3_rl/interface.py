import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TEAM_NAME = "M"


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, tr, info = self.env.step(action)
            total_reward += reward
            done = terminated or tr
            if done:
                truncated = tr
                break
        return obs, total_reward, done, truncated, info


class CnnActorCriticNetwork(nn.Module):
    def __init__(self, output_size):
        super(CnnActorCriticNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.feature_dim = 64 * 11 * 11

        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim, 256), nn.ReLU(), nn.Linear(256, output_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.feature_dim, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor):
        features = self.cnn(x)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        action_logits = self.actor(features)
        return torch.argmax(action_logits, dim=-1)

    def get_action(self, x: torch.Tensor):
        features = self.cnn(x)
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
        action_dim = 5
        self.model = CnnActorCriticNetwork(action_dim)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def act(self, observation: np.ndarray) -> int:
        obs = observation.transpose(2, 0, 1) / 255.0
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        action_idx = self.model.act(obs_tensor)
        return action_idx.item()


class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        env = SkipFrame(env, skip=4)
        super().__init__(env)
