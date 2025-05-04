import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from agent import Agent
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import os
import csv
from utils import *
import numpy as np
from PIL import Image
import cv2
import gc

# THIS MODEL IS FOR TESTING OPEN CV WITH OPENAI GYM SUPER MARIO BROS. IT IS NOT USED IN THE FINAL PROJECT.

MODEL_DESCRIPTION = "Simple Movement + OpenCV Template Matching | Enemy-Aware"
TEMPLATE_DIR = "Enemy Templates"
MATCH_THRESHOLD = 0.7  # Confidence threshold for matchTemplate

def load_templates():
    templates = {}
    for filename in os.listdir(TEMPLATE_DIR):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            name = os.path.splitext(filename)[0]
            path = os.path.join(TEMPLATE_DIR, filename)
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                template = cv2.resize(template, (16, 16))
                templates[name] = template
            else:
                print(f"Warning: Could not load template {filename}")
    return templates

def run(config):
    SHOULD_TRAIN = config["SHOULD_TRAIN"]
    DISPLAY = config["DISPLAY"]
    USE_ENEMY_CHANNEL = config.get("USE_ENEMY_CHANNEL", False)
    NUM_OF_EPISODES = config["NUM_OF_EPISODES"]
    CKPT_SAVE_INTERVAL = config["CKPT_SAVE_INTERVAL"]
    ENV_NAME = config["ENV_NAME"]
    TAG = config["MODEL_TAG"]

    timestamp = get_current_date_time_string()
    model_path = os.path.join("models", TAG, timestamp)
    os.makedirs(model_path, exist_ok=True)

    log_file_path = os.path.join(model_path, "training_log.csv")
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "TotalReward", "Epsilon", "EnemyVisible", "ReachedFlag"])

    render_mode = 'human' if DISPLAY else 'rgb_array'
    env = gym_super_mario_bros.make(ENV_NAME, render_mode=render_mode, apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = apply_wrappers(env)

    obs_shape = env.observation_space.shape  # (4, 84, 84)
    input_dims = (obs_shape[0] + 1, obs_shape[1], obs_shape[2]) if USE_ENEMY_CHANNEL else obs_shape
    agent = Agent(input_dims=input_dims, num_actions=env.action_space.n, use_enemy_channel=USE_ENEMY_CHANNEL)

    templates = load_templates() if USE_ENEMY_CHANNEL else {}

    env.reset()
    env.step(action=0)

    # === Restore from previous checkpoint and epsilon ===
    checkpoint_name = "model_25000_iter.pt"
    checkpoint_path = os.path.join("models", TAG, checkpoint_name)
    start_episode = 0

    if os.path.exists(checkpoint_path):
        print(f"📦 Loading model from: {checkpoint_path}")
        agent.load_model(checkpoint_path)

        try:
            log_path = os.path.join("models", TAG, "training_log.csv")
            with open(log_path, 'r') as f:
                reader = list(csv.DictReader(f))
                if reader:
                    last_entry = reader[-1]
                    agent.epsilon = float(last_entry.get("Epsilon", 1.0))
                    start_episode = int(last_entry.get("Episode", 0)) + 1
                    print(f"🔁 Resuming from episode {start_episode}, epsilon = {agent.epsilon}")
        except Exception as e:
            print(f"Could not restore epsilon/episode: {e}")

    # === Training Loop ===
    for i in range(start_episode, NUM_OF_EPISODES):
        done = False
        state, _ = env.reset()
        total_reward = 0
        prev_x = 0
        enemy_visible = 0

        while not done:
            if USE_ENEMY_CHANNEL:
                a = agent.choose_action((state, enemy_visible))
            else:
                a = agent.choose_action(state)

            # Force always sprint + block left
            if a in [5, 6, 7, 8]:
                a = 3
            elif a == 1:
                a = 3
            elif a == 2:
                a = 4

            new_state, reward, done, truncated, info = env.step(a)

            # Reward shaping
            if info.get("flag_get"):
                print(f"FLAG REACHED at episode {i}")
                reward += 500

            current_x = info.get("x", 0)
            if current_x <= prev_x:
                reward -= 2
            else:
                reward += (current_x - prev_x) * 0.05  # Encourage movement
            prev_x = current_x

            # Enemy detection (OpenCV)
            if USE_ENEMY_CHANNEL and not DISPLAY:
                frame = env.render()
                if frame is not None and isinstance(frame, np.ndarray):
                    try:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        for name, template in templates.items():
                            res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
                            locs = np.where(res >= MATCH_THRESHOLD)
                            for (x, y) in zip(locs[1], locs[0]):
                                enemy_visible = 1
                                print(f"👾 Detected '{name}' at ({x}, {y}) | Match value above {MATCH_THRESHOLD}")
                    except cv2.error as e:
                        print(f"cv2 error: {e}")

            total_reward += reward

            if SHOULD_TRAIN:
                if USE_ENEMY_CHANNEL:
                    agent.store_in_memory((state, enemy_visible), a, reward, (new_state, enemy_visible), done)
                else:
                    agent.store_in_memory(state, a, reward, new_state, done)

                agent.learn()

            state = new_state

        print(f"Episode {i} | Total reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.4f} | Model: {TAG}")

        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            flag_reached = int(info.get("flag_get", False))
            writer.writerow([i, total_reward, agent.epsilon, enemy_visible, flag_reached])

        if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
            agent.save_model(os.path.join(model_path, f"model_{i + 1}_iter.pt"))

    # Final cleanup
    gc.collect()
    agent.save_model(os.path.join(model_path, "final_model.pt"))
    env.close()
