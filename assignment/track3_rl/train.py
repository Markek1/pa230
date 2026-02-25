import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from interface import CnnActorCriticNetwork

LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_BETA = 0.01
VALUE_LOSS_COEF = 0.5
MAX_EPISODES = 2000
UPDATE_TIMESTEPS = 4096
K_EPOCHS = 10
BATCH_SIZE = 64
FRAME_SKIP = 4


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
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


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    gae = 0
    returns = []
    values = values + [next_value]

    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        )
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])

    return returns


def preprocess_obs(obs):
    return obs.transpose(2, 0, 1) / 255.0


def train():
    env = gym.make("CarRacing-v3", continuous=False)
    env = SkipFrame(env, skip=FRAME_SKIP)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Training PPO on CarRacing (Discrete) using {device} with Skip={FRAME_SKIP}..."
    )

    action_dim = env.action_space.n

    agent = CnnActorCriticNetwork(action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR)

    scores = []

    memory_states = []
    memory_actions = []
    memory_log_probs = []
    memory_rewards = []
    memory_dones = []
    memory_values = []

    timestep = 0

    for i_episode in range(MAX_EPISODES):
        state, _ = env.reset()
        state = preprocess_obs(state)
        episode_reward = 0

        while True:
            state_t = (
                torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            )

            action, log_prob, value = agent.get_action(state_t)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            next_state = preprocess_obs(next_state)
            done = terminated or truncated

            memory_states.append(state)
            memory_actions.append(action.item())
            memory_log_probs.append(log_prob.item())
            memory_rewards.append(reward)
            memory_dones.append(done)
            memory_values.append(value.item())

            state = next_state
            episode_reward += reward
            timestep += 1

            if timestep >= UPDATE_TIMESTEPS:
                next_state_t = (
                    torch.as_tensor(next_state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
                _, next_value = agent(next_state_t)
                next_value = next_value.item()

                returns = compute_gae(
                    memory_rewards,
                    memory_values,
                    memory_dones,
                    next_value,
                    GAMMA,
                    GAE_LAMBDA,
                )

                t_states = torch.tensor(
                    np.array(memory_states), dtype=torch.float32
                ).to(device)
                t_actions = torch.tensor(memory_actions, dtype=torch.int64).to(device)
                t_old_log_probs = torch.tensor(
                    memory_log_probs, dtype=torch.float32
                ).to(device)
                t_returns = torch.tensor(returns, dtype=torch.float32).to(device)
                t_values = torch.tensor(memory_values, dtype=torch.float32).to(device)

                t_advantages = t_returns - t_values
                t_advantages = (t_advantages - t_advantages.mean()) / (
                    t_advantages.std() + 1e-8
                )

                dataset_size = len(t_states)
                indices = np.arange(dataset_size)

                for _ in range(K_EPOCHS):
                    np.random.shuffle(indices)

                    for start in range(0, dataset_size, BATCH_SIZE):
                        end = start + BATCH_SIZE
                        idx = indices[start:end]
                        idx_tensor = torch.tensor(idx, dtype=torch.long)

                        mb_states = t_states[idx_tensor]
                        mb_actions = t_actions[idx_tensor]
                        mb_old_log_probs = t_old_log_probs[idx_tensor]
                        mb_returns = t_returns[idx_tensor]
                        mb_advantages = t_advantages[idx_tensor]

                        features = agent.cnn(mb_states)
                        logits = agent.actor(features)
                        new_values = agent.critic(features).squeeze()

                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        new_log_probs = dist.log_prob(mb_actions)
                        dist_entropy = dist.entropy().mean()

                        ratio = torch.exp(new_log_probs - mb_old_log_probs)

                        surr1 = ratio * mb_advantages
                        surr2 = (
                            torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                            * mb_advantages
                        )
                        actor_loss = -torch.min(surr1, surr2).mean()

                        value_loss = F.mse_loss(new_values, mb_returns)

                        loss = (
                            actor_loss
                            + VALUE_LOSS_COEF * value_loss
                            - ENTROPY_BETA * dist_entropy
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                        optimizer.step()

                memory_states = []
                memory_actions = []
                memory_log_probs = []
                memory_rewards = []
                memory_dones = []
                memory_values = []
                timestep = 0

            if done:
                break

        scores.append(episode_reward)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)

        if i_episode % 5 == 0:
            print(
                f"Episode {i_episode}\tScore: {episode_reward:.2f}\tAvg Score: {avg_score:.2f}"
            )

        if avg_score >= 350:
            print(f"Solved in {i_episode} episodes! (Continuing...)")

        if i_episode % 50 == 0:
            torch.save(agent.state_dict(), "./track3_rl/weights.pth")
            print(f"Weights saved to ./track3_rl/weights.pth at episode {i_episode}")

    torch.save(agent.state_dict(), "./track3_rl/weights.pth")
    print("Weights saved to ./track3_rl/weights.pth")
    env.close()


if __name__ == "__main__":
    train()
