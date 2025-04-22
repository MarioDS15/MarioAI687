import csv
import matplotlib.pyplot as plt
import numpy as np

log_file = 'models/Model 2/2025-04-20-22_16_03/training_log.csv'

episodes = []
rewards = []

with open(log_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        episodes.append(int(row["Episode"]))
        rewards.append(float(row["TotalReward"]))

group_size = 50
grouped_episodes = []
avg_rewards = []
best_rewards = []

for i in range(0, len(episodes), group_size):
    chunk = rewards[i:i+group_size]
    if len(chunk) == group_size:
        grouped_episodes.append(episodes[i + group_size // 2])  # midpoint of group
        avg_rewards.append(np.mean(chunk))
        best_rewards.append(np.max(chunk))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(grouped_episodes, avg_rewards, label="Average Reward (per 50 episodes)", marker='o')
plt.plot(grouped_episodes, best_rewards, label="Best Reward (per 50 episodes)", linestyle='--', marker='x')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress: Best and Average Rewards (Grouped Every 50 Episodes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
