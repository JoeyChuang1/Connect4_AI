# Connect4_AI
Engineered Actor-Critic and Rainbow DQN reinforcement learning agents for the Connect4 gym environment,
achieving a 100% win rate against all online bots and perfecting the model’s decision-making. Conducted
performance evaluations from both puzzle and empty states, demonstrating the impact of pre-populated initial
conditions in accelerating RL training efficiency.
This project explores the application of reinforcement learning algorithms—specifically Deep Q-Networks (DQN) and Actor-Critic methods—in multi-agent, sparse reward environments using the game Connect Four as a testbed. Implemented from scratch in Python with PyTorch, NumPy, and Matplotlib, the project investigates the performance, training efficiency, and strategic development of these algorithms. 

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Environment Details](#environment-details)
- [Algorithms Implemented](#algorithms-implemented)
  - [Deep Q-Network (DQN)](#deep-q-network-dqn)
  - [Advanced DQN (Rainbow-inspired)](#advanced-dqn-rainbow-inspired)
  - [Actor-Critic Method](#actor-critic-method)
  - [Key State Buffer](#key-state-buffer)
- [Methodology](#methodology)
- [Experiments and Findings](#experiments-and-findings)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Future Work](#future-work)
- [References](#references)

## Overview

Connect Four offers a balanced complexity for studying reinforcement learning strategies. Unlike computationally intensive methods like Monte Carlo Tree Search (MCTS), this project focuses on classical reinforcement learning algorithms to understand their capabilities and limitations in strategic games without explicit search mechanisms.

## Key Features

- **Deep Q-Network (DQN):** Utilizes convolutional neural networks to approximate Q-values, enabling the agent to learn optimal policies through experience replay.

- **Advanced DQN (Rainbow-inspired):** Enhances the standard DQN with prioritized experience replay and noisy linear layers to improve sample efficiency and exploration.

- **Actor-Critic Method:** Combines policy-based and value-based approaches, where the actor selects actions and the critic evaluates them, facilitating stable and efficient learning.

- **Key State Buffer:** Introduces a buffer that stores significant states (e.g., winning or losing positions) to propagate reward signals more effectively in sparse reward settings.

## Environment Details

The project employs the [PettingZoo](https://www.pettingzoo.ml/) library to simulate the Connect Four environment:

- **Observation Space:** A 2x6x7 tensor representing the board state for both players.

- **Action Space:** Discrete actions ranging from 0 to 6, corresponding to the columns where a token can be dropped.

- **Reward Structure:** +1 for a win, -1 for a loss, and 0 for a draw or non-terminal move.

## Algorithms Implemented

### Deep Q-Network (DQN)

The DQN model uses a convolutional neural network to approximate the Q-function. The architecture includes:

- Two convolutional layers with 2x2 filters to extract spatial features from the board state.

- A flattening layer to convert the 2D feature maps into a 1D feature vector.

- A fully connected layer with 128 units followed by a ReLU activation function.

- An output layer with 7 units corresponding to the action space (columns 0-6).

Experience replay is utilized with a buffer that samples batches of 128 experiences per episode to stabilize training.

### Advanced DQN (Rainbow-inspired)

This version incorporates elements from the Rainbow DQN architecture to enhance learning efficiency:

- **Prioritized Experience Replay (PER):** Transitions with higher temporal-difference (TD) errors are sampled more frequently, focusing learning on more informative experiences.

- **Noisy Linear Layers:** Introduces stochasticity in the network's weights to encourage exploration without relying solely on ε-greedy strategies.

These enhancements aim to improve sample efficiency and accelerate the learning process.

### Actor-Critic Method

The Actor-Critic model consists of two components:

- **Actor:** Selects actions based on a policy derived from the current state.

- **Critic:** Evaluates the chosen actions by estimating the value function.

The network architecture mirrors that of the DQN, with convolutional layers followed by fully connected layers. The output layer uses a Softmax activation function to produce a probability distribution over actions.

The model updates its parameters using the policy gradient theorem and the advantage function, which measures the relative value of an action compared to the average.

### Key State Buffer

To address the challenge of sparse rewards, a Key State Buffer is implemented:

- **Purpose:** Stores states that result in significant rewards (+1 or -1) to reinforce learning from critical game moments.

- **Mechanism:** Acts as a prioritized buffer, ensuring that impactful experiences are revisited more frequently during training.

This approach helps propagate reward signals backward through the network, facilitating the discovery of effective strategies.

## Methodology

The study follows a structured approach to evaluate and compare the algorithms:

1. **Training Against Random Agents:** Each model is initially trained against a random agent to establish baseline performance.

2. **Self-Play Training:** Agents are then trained against copies of themselves to observe the evolution of strategies over time.

3. **Cross-Algorithm Matches:** Different algorithms are pitted against each other to assess comparative strengths and weaknesses.

4. **Key State Buffer Implementation:** The impact of the key state buffer on learning efficiency and performance is analyzed.

## Experiments and Findings

- **Learning Capability:** Both DQN and Actor-Critic models successfully learn to play Connect Four, demonstrating the viability of classical reinforcement learning methods in such environments.

- **First-Move Advantage:** Consistent with game theory, the first player often holds an advantage, which is reflected in the training outcomes.

- **Enhanced Learning with Key State Buffer:** Incorporating a key state buffer accelerates learning by effectively propagating reward signals, especially in sparse reward scenarios.

## Usage

To run the experiments, execute the corresponding Jupyter notebooks:

- `connect4_dqn.ipynb`: Implementation and training of the standard Deep Q-Network.

- `connect4_dqn_rainbowish.ipynb`: Enhanced DQN with prioritized experience replay and noisy layers.

- `connect_4_rl.ipynb`: Implementation of the Actor-Critic algorithm.

Ensure that all dependencies are installed before running the notebooks.

## Dependencies

Make sure the following Python packages are installed:

- `numpy`

- `torch`

- `matplotlib`

- `pettingzoo`

You can install them using pip:

```bash
pip install numpy torch matplotlib pettingzoo

Future Work
The project aims to extend the current methodologies to more complex games like chess, requiring additional computational resources and potentially more sophisticated algorithms to handle increased complexity and sparser rewards.

References
Mnih et al., "Human-level control through deep reinforcement learning," Nature, 2015.

Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning," 2018.

Schaul et al., "Prioritized Experience Replay," 2015.

Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments," 2017.

Silver et al., "Mastering the game of Go with deep neural networks and tree search," Nature, 2016.

Balduzzi et al., "Open-ended learning in symmetric
