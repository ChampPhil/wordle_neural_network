# Wordle Project

A Wordle game implementation with two play modes: **Human mode** (manual play) and **AI mode** (neural network–powered solver).

## Overview

This project implements the classic Wordle word-guessing game. Players have 6 guesses to identify a 5-letter secret word. Each guess receives color-coded feedback: gray (letter not in word), yellow (letter in word, wrong position), and green (letter correct and in position).

## Features

- **Human mode**: Play Wordle manually by typing your guesses.
- **AI mode**: Watch a reinforcement learning agent (Maskable PPO) solve Wordle automatically.
- Uses the official Wordle word lists for valid guesses and secret words.

## Requirements

- Python 3.x
- Dependencies: `stable-baselines3`, `shimmy>=2.0`, `torch`, `sb3-contrib`, `colorama`, `gymnasium`, `numpy`

Install with:

```bash
pip install stable-baselines3 shimmy>=2.0 torch sb3-contrib colorama gymnasium numpy
```

## Running the Game

```bash
python wordle.py
```

On startup, choose:
- **1** for manual (human) mode
- **2** for AI mode

## Project Structure

| File | Description |
|------|-------------|
| `wordle.py` | Main game logic and entry point |
| `wordle_env.py` | Gymnasium environment for the Wordle RL agent |
| `wordle_utils.py` | Utilities for feedback, state encoding, and word conversion |
| `wordle_secret_words.py` | Loads the secret word list |
| `valid_wordle_guesses.py` | Loads the valid guess list |
| `data/` | Word lists (`wordle_secret_words.txt`, `wordle_valid_guesses.txt`) |

## Neural Network & Reinforcement Learning

The AI solver is a **Maskable PPO** (Proximal Policy Optimization) agent trained with reinforcement learning. The `model.zip` file was created by training the neural network in a separate Google Colab notebook, which you can view here:  (https://colab.research.google.com/drive/1hQWDi_2_-kbC6l1argb4yZsxI5aJncQP)

Ensure the trained model files are present in the project directory before running in AI mode.

### What is Reinforcement Learning?

**Reinforcement learning (RL)** is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. Unlike supervised learning (where the model learns from labeled examples) or unsupervised learning (where it finds patterns in unlabeled data), RL learns from **trial and error** guided by **rewards**.

The core loop works like this:

1. The agent observes the current **state** (e.g., the board after previous guesses).
2. It selects an **action** (e.g., guessing a word).
3. The environment returns a **reward** (positive for progress, negative for failure) and a new state.
4. The agent updates its **policy**—the mapping from states to actions—to maximize cumulative reward over time.

In Wordle, the agent learns which words to guess by playing many games. It receives rewards for narrowing the search space and for winning, and penalties for losing. Over thousands of episodes, it discovers strategies that lead to higher total reward. **PPO (Proximal Policy Optimization)** is the specific algorithm used: it updates the policy in small, stable steps to avoid catastrophic changes that could undo previous learning.

### Why Maskable PPO?

Wordle has a large action space (2,313 possible guesses) but at any given moment only a subset of words are *valid*—i.e., consistent with the feedback so far. **Maskable PPO** from `sb3-contrib` uses **action masking**: invalid actions are masked out so the policy never selects them. This dramatically improves sample efficiency and guarantees the agent only guesses words that could still be the secret word.

### State Representation (Observation)

The agent observes a **288-dimensional** vector encoding the current game state:

| Component | Size | Description |
|-----------|------|-------------|
| `absent_letters_vector` | 26 | Binary: which letters are proven *not* in the word (gray feedback) |
| `required_letter_nums_and_positions_vector` | 130 | Binary: which letter–position pairs are confirmed correct (green feedback). 26 letters × 5 positions |
| `required_letter_nums_and_wrong_positions_vector` | 130 | Binary: which letter–position pairs are known wrong (yellow feedback) |
| `normalized_guesses_remaining` | 1 | `remaining_guesses / max_guesses` |
| `valid_count_feature` | 1 | Fraction of secret words still consistent with feedback |

This encoding is inspired by [Towards Data Science: Finding the Best Wordle Opener](https://towardsdatascience.com/finding-the-best-wordle-opener-with-machine-learning-ce81331c5759) and [Andrew Kho's Wordle Solver](https://andrewkho.github.io/wordle-solver/).

### Reward Structure

- **Win**: `10 + (6 - num_guesses_made) × 2` — bonus for solving in fewer guesses
- **Loss** (out of guesses): `-10`
- **Non-terminal** (guess made, game continues): `reduction_ratio × 2 - 0.5`, where `reduction_ratio = 1 - (valid_words_after / valid_words_before)`. The agent is rewarded for guesses that narrow the search space.

### Network Architecture

The policy uses an MLP (multi-layer perceptron) with shared feature extraction and separate heads for policy (π) and value (V):

- **Policy network (π)**: `[256, 128]` hidden layers
- **Value network (V)**: `[256, 128]` hidden layers
- **Activation**: LeakyReLU
- **Optimizer**: Adam

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Total timesteps | 30,000 |
| Learning rate | Warmup–cosine schedule: 5e-4 peak, 2% warmup, 5e-6 minimum |
| Batch size | 128 |
| Clip range | 0.2 |
| Entropy coefficient | 0.001 |

**Callbacks:**
- **MaskableEvalCallback**: Evaluate every 10,000 steps, 100 episodes per eval
- **StopTrainingOnNoModelImprovement**: Early stopping after 10 evals without improvement (min 20 evals first)

### Environment

The agent is trained in a custom **Gymnasium** `WordleEnv`:
- **Action space**: `Discrete(2313)` — one action per secret word
- **Observation space**: `Box(shape=(288,), low=0, high=1, dtype=float32)`
- Wrapped with `ActionMasker` to supply the valid-action mask to MaskablePPO

