# Proximal Policy Optimization (PPO) for Highway Environment

## Overview

This project is an implementation of the Proximal Policy Optimization (PPO) algorithm for training an agent to navigate in the "highway-fast" environment, which is provided by the "highway_env" module. PPO is a reinforcement learning algorithm used to train agents to perform tasks in environments with high-dimensional state spaces and continuous action spaces.

The project includes the following main components:

1. Importing Required Libraries:
   - The project starts by importing various Python libraries, including Gym, Ray, highway_env, PyTorch, Matplotlib, and others. These libraries are used for reinforcement learning, deep learning, and visualization.

2. Environment Setup:
   - The Gym environment "highway-fast-v0" is set up for training the agent. The environment's configuration is modified to use grayscale observations, reduce the observation size, and set the policy frequency.

3. Neural Network Models:
   - Two neural network models are defined: `actor` and `critic`. These models are used to approximate the policy (actor) and the value function (critic) for the PPO algorithm. The actor model outputs action probabilities, and the critic model estimates state values.

4. PPO Algorithm Implementation:
   - The PPO class is implemented to handle the training and learning process. The PPO algorithm consists of the following steps:
     - Rollout: Gather data by interacting with the environment for multiple episodes.
     - Compute Returns and Advantages: Calculate returns and advantages for policy optimization.
     - Evaluate: Compute action probabilities and value estimates using the actor and critic networks.
     - Update the Policy: Update the actor and critic networks based on the PPO loss functions.
     - Save Model: Periodically save the actor and critic model weights.

5. Training Loop:
   - The PPO class is instantiated, and the agent is trained in a loop. Training continues until a maximum number of time steps are reached. At each iteration, data is collected, and the policy and value networks are updated to improve the agent's performance.

6. Model Saving:
   - The trained actor and critic models are saved to disk at regular intervals to allow for continued training or evaluation later.

## Dependencies

Before running this code, ensure you have the following libraries and dependencies installed:

- Gym
- Ray
- highway_env
- PyTorch
- Matplotlib
- NumPy
- PIL (Pillow)

Additionally, make sure you have the "highway-fast" Gym environment properly installed.

## Usage

To use this project, you can follow these steps:

1. Install the required libraries and dependencies.
2. Run the provided code in a Python environment that supports Gym and PyTorch.
3. The agent will undergo training, and the actor and critic models will be periodically saved.

The trained models can be loaded later for evaluation or further training.

This project provides an implementation of the PPO algorithm for training an agent in a highway navigation environment. It is designed to serve as a foundation for reinforcement learning experiments in complex environments. Further improvements and customizations can be made to suit specific tasks and domains.
