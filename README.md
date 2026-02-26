# PA230 — Reinforcement Learning Agents (Tournament)

Coursework repository for the PA230 tournament. It contains my implementations of deep RL agents in Python + PyTorch, evaluated in Gymnasium environments:

- **LunarLander-v3 (discrete)** — value-based + policy-gradient approaches
- **CarRacing-v3 (discrete, from pixels)** — CNN-based policy trained from raw frames

## What I implemented

- **Track 1 (`assignment/track1_dqn`)**: Double DQN-style training loop with replay buffer, target network + soft updates (Polyak), epsilon-greedy exploration, gradient clipping.
- **Track 2 (`assignment/track2_pg`)**: PPO-style actor-critic with clipped objective, GAE advantage estimates, entropy bonus, value baseline.
- **Track 3 (`assignment/track3_rl`)**: PPO with a CNN policy/value network for $96\times96$ pixel observations, observation preprocessing, and a simple frame-skip wrapper.

The trained checkpoints used for evaluation live next to each track as `weights.pth`.

## Repo layout

- `assignment/` — tournament scaffold (evaluation script, interface contract, and my track solutions)
    - `assignment/eval.py` — evaluator used to sanity-check submissions
    - `assignment/example_submission/` — example of the required submission folder structure
- `practise/` — small side experiments (e.g., tabular value iteration)

## Quickstart

Dependencies are listed in `assignment/requirements.txt`.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r assignment/requirements.txt
```

## Run evaluation (loads `interface.py` + `weights.pth`)

From the repository root:

```sh
python assignment/eval.py --track <TRACK_NUM> --load_path assignment/<TRACK_PATH> --eps 10
```

## Train (writes new `weights.pth`)

```sh
python assignment/<TRACK_PATH>/train.py
```


- The tournament submission format is: an `agent/` directory containing `interface.py` and `weights.pth` (see `assignment/example_submission/`).
- For a short write-up of design choices, see `assignment/REPORT.md`.
