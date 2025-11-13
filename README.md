# Exploring Reinforcement Learning through Super Mario Bros

This repository contains a research project that analyzes how different reinforcement learning strategies perform in a controlled Super Mario Bros environment. The project evaluates several independently trained models to measure the individual impact of reward shaping, heuristic feedback, and computer vision on agent performance. Each model is trained separately to isolate the effect of each technique.

The full research report is included in the repository.

---

## Project Goals

- Investigate how Double DQN performs in a classic platformer environment.
- Compare multiple agent designs to measure each method's solo impact.
- Evaluate how reward shaping, heuristic logic, and computer vision influence learning.
- Identify reinforcement learning limitations that emerge in dynamic game environments.

---

## Model Overview

The project contains four models. Each model builds on the same Double DQN foundation but introduces a specific modification to study its isolated effect.

### Model 1: Basic Reward Driven Agent
A simple Double DQN agent trained with a minimal reward structure. The agent receives rewards for moving right, completing the level, and negative rewards when dying. This model serves as a baseline for measuring improvements in later models.

### Model 2: Stuck Detection with Negative Reinforcement
Adds a heuristic to detect when the agent stops making forward progress. If the agent remains in the same area for too long, it receives a penalty. This encourages the agent to escape loops and attempt new actions, improving exploration compared to Model 1.

### Model 3: Reward Shaping with Enemy Detection
Introduces computer vision using template matching to detect enemies on the screen. The agent receives a reward only when it successfully jumps over an enemy and survives. The reward requires Mario to pass above the enemy in the y axis and then surpass it in the x axis before landing safely. This improves the agent’s ability to avoid hazards.

### Model 4: Enhanced Reward Logic (Planned or In Progress)
A further extension of Model 3 that refines the reward logic and expands enemy detection. This model is designed to provide more consistent feedback and enable deeper interactions with the environment. If implemented in the future, it will expand the feature set established in Model 3.

---

## Results Summary

- Model 1 and Model 3 achieved similar learning rates but Model 3 reached higher stability due to visual enemy detection.
- Model 2 improved escape behavior in situations where the agent previously stalled.
- Each model demonstrated unique strengths and weaknesses, confirming that small architectural changes can significantly alter performance.
- Reinforcement learning faced challenges such as sparse rewards, feature blindness, and limited generalization across more complex environments.

---

## Limitations Observed

- Sparse rewards led to local optima.
- The agent failed to detect hidden mechanics such as the frame rule.
- Visual detection required significant preprocessing and debugging.
- Training was long and resource intensive due to large replay buffers and 100,000 episode requirements.

---

## Future Work

- Expand the action space to include backward movement and more nuanced controls.
- Add logic for later game enemies with more complex behavior patterns.
- Build a modular framework where a specialized agent is loaded for each level.
- Explore intrinsic motivation or curiosity-based rewards for richer exploration.

---

## Report

The complete research report can be found in:

**`Exploring Reinforcement Learning through Super Mario Bros.pdf`**

---

## Acknowledgements

The base environment for this project was adapted from the open-source repository Super-Mario-Bros-RL by Sourish07.
