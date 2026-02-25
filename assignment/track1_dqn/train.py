import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

from interface import QNetwork

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.97
TAU = 0.005
LR = 1e-3
MEMORY_SIZE = 100000
MIN_MEMORY_SIZE = 1000
MAX_EPISODES = 3000
MAX_STEPS = 1000
TARGET_UPDATE_FREQ = 1
UPDATE_FREQ = 4


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.epsilon = EPS_START

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32)
            return self.policy_net.act(state_t).item()

    def update(self, buffer):
        if len(buffer) < MIN_MEMORY_SIZE:
            return None

        states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

        states = torch.tensor(states)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards).unsqueeze(1)
        next_states = torch.tensor(next_states)
        dones = torch.tensor(dones).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)

            target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update()

        # self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

        return loss.item()

    def soft_update(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                TAU * policy_param.data + (1.0 - TAU) * target_param.data
            )


def train():
    env = gym.make("LunarLander-v3")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Training...")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    buffer = ReplayBuffer(MEMORY_SIZE)

    scores = []

    for i_episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, int(terminated))

            state = next_state
            episode_reward += reward

            if t % UPDATE_FREQ == 0:
                agent.update(buffer)

            if done:
                break

        scores.append(episode_reward)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        agent.epsilon = max(EPS_END, agent.epsilon * EPS_DECAY)

        if i_episode % 10 == 0:
            print(
                f"Episode {i_episode}\tScore: {episode_reward:.2f}\tAvg Score: {avg_score:.2f}\tEpsilon: {agent.epsilon:.2f}"
            )

        # if avg_score >= 250:  # Solved threshold slightly higher than req
        #     print(f"Solved in {i_episode} episodes! (Continuing training...)")
            # break

    torch.save(agent.policy_net.state_dict(), "./track1_dqn/weights.pth")
    print("Weights saved to ./track1_dqn/weights.pth")
    env.close()


if __name__ == "__main__":
    train()
