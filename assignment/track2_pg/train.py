import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from interface import ActorCriticNetwork

LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_BETA = 0.01
VALUE_LOSS_COEF = 0.5
MAX_EPISODES = 2000
UPDATE_TIMESTEPS = 2048
K_EPOCHS = 10
BATCH_SIZE = 64


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


def train():
    env = gym.make("LunarLander-v3")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Training PPO...")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCriticNetwork(state_dim, action_dim)
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
        episode_reward = 0

        while True:
            state_t = torch.as_tensor(state, dtype=torch.float32)
            action, log_prob, value = agent.get_action(state_t)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
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
                next_state_t = torch.as_tensor(next_state, dtype=torch.float32)
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

                t_states = torch.tensor(np.array(memory_states), dtype=torch.float32)
                t_actions = torch.tensor(memory_actions, dtype=torch.int64)
                t_old_log_probs = torch.tensor(memory_log_probs, dtype=torch.float32)
                t_returns = torch.tensor(returns, dtype=torch.float32)
                t_values = torch.tensor(memory_values, dtype=torch.float32)

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

                        mb_states = t_states[idx]
                        mb_actions = t_actions[idx]
                        mb_old_log_probs = t_old_log_probs[idx]
                        mb_returns = t_returns[idx]
                        mb_advantages = t_advantages[idx]

                        features = agent.base(mb_states)
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

        if i_episode % 10 == 0:
            print(
                f"Episode {i_episode}\tScore: {episode_reward:.2f}\tAvg Score: {avg_score:.2f}"
            )

    torch.save(agent.state_dict(), "./track2_pg/weights.pth")
    print("Weights saved to ./track2_pg/weights.pth")
    env.close()


if __name__ == "__main__":
    train()
