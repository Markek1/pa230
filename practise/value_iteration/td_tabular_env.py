from datetime import datetime
import gymnasium as gym
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from functools import cached_property
from collections import namedtuple


"""
    Renders a heatmap of policy values
"""


def render_heatmap(frame, values, rows, cols, env_name, algo_name, ax=None):
    cell_height = frame.shape[0] // rows
    values = np.array(values).reshape((rows, cols))

    if ax is None:
        ax = plt.gca()
    cmap = cm.get_cmap("Greys")

    im = ax.imshow(values, cmap=cmap, extent=[0, cols, 0, rows])
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Values", rotation=-90, va="bottom")

    for i in range(rows):
        for j in range(cols):
            ax.text(
                j + 0.5,
                rows - i - 0.5,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                color="red",
                fontsize=cell_height * 0.2,
            )

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(1, values.shape[1] + 1), minor=True)
    ax.set_yticks(np.arange(1, values.shape[0] + 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(f"Heatmap of {algo_name} value function on {env_name}.")


"""
    Renders rows*cols arrows on the frame


    movement_dirs are \in {"LEFT", "RIGHT"}, etc., translated from the action
    indices to the real actions. Expected shape is (rows * cols, )

    rows/cols describe the number of states in each dimension of the rendered
    map

"""


def render_actions(frame, movement_dirs, rows, cols, env_name, algo_name, ax=None):

    if ax is None:
        ax = plt.gca()

    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    nS = len(movement_dirs)

    height, width, _ = frame.shape

    frame = rgb2gray(frame)

    env_image = ax.imshow(frame, cmap="gray")

    cell_height = height // rows
    cell_width = width // cols

    dir_to_arrow = {
        "LEFT": (-0.25 * cell_width, 0),
        "DOWN": (0, 0.25 * cell_height),
        "RIGHT": (0.25 * cell_width, 0),
        "UP": (0, -0.25 * cell_height),
    }

    for state in range(nS):

        row = state // cols
        col = state % cols

        # get arrow from action description
        direction = dir_to_arrow[movement_dirs[state]]

        center_x = col * cell_width + cell_width // 2
        center_y = row * cell_height + cell_height // 2

        dx, dy = direction

        # draw arrow in the current cell, pointing in the dir of optimal action
        ax.arrow(
            center_x,
            center_y,
            dx,
            dy,
            color="red",
            head_width=cell_width * 0.2,
            head_length=cell_height * 0.2,
        )

    ax.set_title(f"{env_name} with {algo_name} policy actions")
    return ax


class TabularWrapper:
    """
    Environment wrapper working for CliffWalk and FrozenLake environments that
    provides the following functionality:

    1) Access to environment dynamics and other miscellaneous environment
    information, number of states, actions. See properties/methods:
        - num_states
        - num_actions
        - dynamics_tensor
        - reward_tensor
        - get_transition()
        - get_reward()

    2) Support for rendering Policy objects via `render_policy()`.
    """

    def __init__(self, env, max_samples=-1):
        """
        `max_samples` specify the number of `step()` calls you are able to
        make in this environment. The default value of -1 signals unlimited
        sampling from this environment.
        """

        self.env = env

        # name of the env instance
        self.name = self.env.unwrapped.spec.id

        self.steps = 0
        self.ep_step = 0
        self.episodes = 0
        self.max_steps = max_samples

        self.reset_done = False

        self._initialize_dynamics()

    """
        ENV INTERFACE
    """

    @cached_property
    def num_states(self):
        """
        Returns the number of states in the environment.
        The states are always indices from 0 to num_states - 1.
        """
        return self.env.observation_space.n

    @cached_property
    def num_actions(self):
        """
        Returns the number of actions in the environment.
        """
        return self.env.action_space.n

    def get_transition(self, state, act, succ):
        """
        Returns p(succ | state, act)
        """
        return self.dynamics_tensor[state, act, succ]

    def get_reward(self, state, act, succ):
        """
        Returns r(state, act, succ)
        """
        return self.reward_tensor[state, act, succ]

    def render(self):
        """
        Renders the environment.
        """
        return self.env.render()

    def render_policy(self, policy, label=""):
        """
        Accepts a policy object that implements `play(s): S -> A` and `value(s): S -> R` methods.

        Renders a figure contatining arrows representing the actions played by `policy.play(s)`
        and a heatmap of state values from `policy.value(s)`.
        """
        if not self.reset_done:
            self.reset()

        # Switch env to rgb_array mode to render one frame
        prev_mode = self.env.unwrapped.render_mode
        self.env.unwrapped.render_mode = "rgb_array"
        frame = self.env.render()
        self.env.unwrapped.render_mode = prev_mode

        # Get values of states
        nS = self.num_states
        state_values = [policy.raw(obs) for obs in range(nS)]

        r, c = self._map_dimensions()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Translate action indices to movement directions (the envs differ in the mapping)
        direction_map = self._get_direction_mapping()
        action_directions = [direction_map[policy.play(obs)] for obs in range(nS)]

        render_actions(frame, action_directions, r, c, self.get_name(), label, ax1)
        render_heatmap(frame, state_values, r, c, self.get_name(), label, ax2)

        plt.tight_layout()
        plt.show()

    def step(self, action):
        return self.env.step(action)

    def reset(self, *args, **kwargs):
        self.reset_done = True
        return self.env.reset(*args, **kwargs)

    def clear_stats(self):
        self.steps = 0
        self.ep_step = 0
        self.episodes = 0

        self.episode_rewards = []
        self.episode_lengths = []

    def get_name(self):
        return self.name

    # initialize dynamics and reward tensors from gym data
    def _initialize_dynamics(self):
        if not hasattr(self.env.unwrapped, "P"):
            raise AttributeError(
                "Gym environment passed to the TabularWrapper",
                "does not provide dynamics.",
            )

        nS, nA = self.num_states, self.num_actions

        reward_tensor = np.zeros((nS, nA, nS))
        dynamics_tensor = np.zeros((nS, nA, nS))

        gym_matrix = self.env.unwrapped.P

        # certain states should be absorbing, but the dynamics in P do not match
        # fix this manually here
        absorbing_states = set([47]) if "CliffWalking" in self.name else set([nS - 1])

        # this will store all non-terminal states, so we can explore starts
        # from them later.
        nonterminal_states = []

        for s in range(nS):
            terminal = True
            for a in range(nA):
                if s in absorbing_states:
                    dynamics_tensor[s, a, s] = 1.0

                else:
                    for p, succ, r, _ in gym_matrix[s][a]:
                        dynamics_tensor[s, a, succ] += p
                        reward_tensor[s, a, succ] += r

                        if succ != s:
                            terminal = False

            if not terminal:
                nonterminal_states.append(s)

        self.dynamics_tensor = dynamics_tensor
        self.reward_tensor = reward_tensor
        self.nonterminal = nonterminal_states
        self.init_distr = self.env.unwrapped.initial_state_distrib

    # get number of rows/columns in the rendered map
    def _map_dimensions(self):
        height, width = 0, 0
        if "FrozenLake" in self.name:
            nS = self.num_states
            height = int(sqrt(nS))
            width = height
        elif "CliffWalking" in self.name:
            height = 4
            width = 12

        return height, width

    # translate action index to the actual movement direction
    def _get_direction_mapping(self):
        action_dict = {}
        if "FrozenLake" in self.name:
            action_dict = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

        elif "CliffWalking" in self.name:
            action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
        return action_dict


def value_iteration(env, gamma=0.9, theta=1e-6):
    v = np.zeros(env.num_states)

    while True:
        delta = 0
        v_new = np.zeros(env.num_states)

        for s in range(env.num_states):
            future_values = env.reward_tensor[s] + gamma * v
            q_values = np.sum(env.dynamics_tensor[s] * future_values, axis=1)
            v_new[s] = np.max(q_values)
            delta = max(delta, np.abs(v_new[s] - v[s]))

        v = v_new

        if delta < theta:
            break

    return v


def extract_policy_vi(env, v, gamma=0.9):
    policy_map = np.zeros(env.num_states, dtype=int)

    for s in range(env.num_states):
        future_values = env.reward_tensor[s] + gamma * v
        q_values = np.sum(env.dynamics_tensor[s] * future_values, axis=1)

        policy_map[s] = np.argmax(q_values)

    return policy_map


"""

    Try implementing Q-learning and SARSA as an individual exercise. Instead of
    using the dynamics and reward functions that were necessary for VI, these
    algorithms learn only through interaction with the environment via the
    `step()` function.

    Below is a skeleton for temporal-difference learning you can edit.

    Try training the algorithms (for example for 100000 steps) on the
    CliffWalking and FrozenLake envs.

    Try modifying the lr/epsilon schedules, visualizing the different learned
    policies.

    Questions to think about:

    1) How does the choice (or schedule) of epsilon and LR affect learning?

    2) Which environment is the hardest to learn on, and why? Can you think of
    any ways to improve learning on that environment?

    3) Compare the learned policy of QL and SARSA on CliffWalking.
        Do they differ? Why?

"""


def td_example(env, steps, seed=42):
    # Example starting hyperparameters
    lr = 0.025
    eps = 0.2
    gamma = 0.99

    # Initialize q-values to zero
    q_values = np.zeros((env.num_states, env.num_actions))

    # Seed the env once before training
    state, info = env.reset(seed=seed)
    np.random.seed(seed)

    for i in range(max_steps):

        # TODO: Choose action via epsilon-greedy policy
        if np.random.random() < eps:
            action = env.env.action_space.sample()
        else:
            action = np.argmax(q_values[state])

        # Take a step in the env
        next_state, reward, terminated, truncated, _ = env.step(action)
        # Signals episode end
        done = terminated or truncated

        # TODO construct learning target, don't bootsrap in final state
        # Q-learning target: r + gamma * max_a' Q(s', a')
        best_next_q = np.max(q_values[next_state])
        td_target = reward + (not done) * gamma * best_next_q

        # Update the q-values
        q_values[state, action] += lr * (td_target - q_values[state, action])
        state = next_state

        if done:
            state, info = env.reset()

    # reset env to init state for nicer visualization
    env.reset()
    return q_values


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)  # SmallLake
    # env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    # env = gym.make("CliffWalking-v1")

    env = TabularWrapper(env)

    max_steps = 200000
    q_values = td_example(env, max_steps, seed=42)

    # Extract greedy policy from Q-values
    # Policy: pi(s) = argmax_a Q(s, a)
    # Value: V(s) = max_a Q(s, a)

    best_actions_td = np.argmax(q_values, axis=1)
    state_values_td = np.max(q_values, axis=1)

    Policy = namedtuple("Policy", ["play", "raw"])
    policy_obj = Policy(
        play=lambda s: best_actions_td[s], raw=lambda s: state_values_td[s]
    )

    env.render_policy(policy_obj, label="TD (Q-Learning) Policy")
