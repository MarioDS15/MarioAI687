import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import Agent
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import os
import csv
from utils import *

MODEL_DESCRIPTION = "Right Only | No Pipe Logic"

def run(config):
    # === CONFIGURATION ===
    SHOULD_TRAIN = config["SHOULD_TRAIN"]
    DISPLAY = config["DISPLAY"]
    NUM_OF_EPISODES = config["NUM_OF_EPISODES"]
    CKPT_SAVE_INTERVAL = config["CKPT_SAVE_INTERVAL"]
    ENV_NAME = config["ENV_NAME"]
    USE_ENEMY_CHANNEL = config.get("USE_ENEMY_CHANNEL", False)
    TAG = config["MODEL_TAG"]

    # === FILESETUP ===
    timestamp = get_current_date_time_string()
    model_path = os.path.join("models", TAG, timestamp)
    os.makedirs(model_path, exist_ok=True)

    log_file_path = os.path.join(model_path, "training_log.csv")
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "TotalReward", "Epsilon"])

    # === ENV SETUP ===
    env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)

    obs_shape = env.observation_space.shape  # (4, 84, 84)
    input_dims = (obs_shape[0] + 1, obs_shape[1], obs_shape[2]) if USE_ENEMY_CHANNEL else obs_shape
    agent = Agent(input_dims=input_dims, num_actions=env.action_space.n, use_enemy_channel=USE_ENEMY_CHANNEL)

    # === WARM-UP ===
    env.reset()
    env.step(action=0)

    # === TRAINING LOOP ===
    for i in range(NUM_OF_EPISODES):
        done = False
        state, _ = env.reset()
        total_reward = 0
        enemy_visible = 0  # default for compatibility

        while not done:
            if USE_ENEMY_CHANNEL:
                a = agent.choose_action((state, enemy_visible))
            else:
                a = agent.choose_action(state)

            new_state, reward, done, truncated, info = env.step(a)

            total_reward += reward

            if SHOULD_TRAIN:
                if USE_ENEMY_CHANNEL:
                    agent.store_in_memory((state, enemy_visible), a, reward, (new_state, enemy_visible), done)
                else:
                    agent.store_in_memory(state, a, reward, new_state, done)

                agent.learn()

            state = new_state

        print(f"Episode {i} | Total reward: {total_reward} | Epsilon: {agent.epsilon} | Model: {TAG}")

        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, total_reward, agent.epsilon])

        if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
            agent.save_model(os.path.join(model_path, f"model_{i + 1}_iter.pt"))
    import gc
    gc.collect()
    agent.save_model(os.path.join(model_path, "final_model.pt"))
    env.close()
