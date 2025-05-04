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
from pathlib import Path

MODEL_DESCRIPTION = "Simple Movement + OpenCV Template Matching | Enemy-Aware"
TEMPLATE_DIR = "Enemy Templates"
MARIO_TEMPLATE_DIR = Path("Mario Templates")
MATCH_THRESHOLD = 0.7
VERBOSE_DETECTION = True
MARIO_MATCH_THRESHOLD = 0.7

def load_templates_from(folder_path):
    """Load all image templates from a directory."""
    templates = {}
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return templates

    for file in folder.glob("*.png"):
        name = file.stem
        template = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if template is not None:
            template = cv2.resize(template, (20, 20))
            #template = cv2.GaussianBlur(template, (5, 5), 0.5)
            templates[name] = template
        else:
            print(f"Could not load {file.name}")
    for file in folder.glob("*.jpg"):
        name = file.stem
        template = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if template is not None:
            template = cv2.resize(template, (16, 16))
            #template = cv2.GaussianBlur(template, (3, 3), 0.5)
            templates[name] = template
        else:
            print(f"Could not load {file.name}")
    print("Mario templates loaded:")
    for name in templates.keys():
        if VERBOSE_DETECTION:
            print(f" - {name}")
    return templates
from PIL import Image
from datetime import datetime

DEBUG_MARIO_DIR = Path("mario_detection")
DEBUG_MARIO_DIR.mkdir(exist_ok=True)

def save_debug_frame_with_box(frame, box_coords, label="Mario", match_val=None):
    x, y, w, h = box_coords
    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

    text = f"{label} ({match_val:.2f})" if match_val else label
    cv2.putText(annotated, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    filename = f"debug_mario_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    save_path = DEBUG_MARIO_DIR / filename

    Image.fromarray(annotated).save(save_path)
    print(f"Screenshot saved: {save_path}")


def detect_enemies_and_mario(frame, enemy_templates, mario_templates):
    enemy_visible = 0
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Enemy detection
    for name, template in enemy_templates.items():
        res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= MATCH_THRESHOLD)
        frame_h, frame_w = frame_gray.shape
        cell_w = frame_w // 2
        cell_h = frame_h // 4
        bottom_right_x_min = cell_w
        bottom_right_y_min = 3 * cell_h  # 4th row starts at 3 * cell_h
        
        for (x, y) in zip(locs[1], locs[0]):
            if x >= bottom_right_x_min and y >= bottom_right_y_min:
                enemy_visible = 1
                if VERBOSE_DETECTION:
                    print(f"Enemy Detected '{name}' in BOTTOM-RIGHT at ({x}, {y}) | Match ≥ {MATCH_THRESHOLD}")


    # Mario detection
    for name, template in mario_templates.items():
        res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= MARIO_MATCH_THRESHOLD)
        h, w = template.shape[:2]
        for (x, y) in zip(locs[1], locs[0]):
            if VERBOSE_DETECTION:
                print(f"🧍 Mario detected at ({x}, {y}) from template '{name}' | Match ≥ {MARIO_MATCH_THRESHOLD}")
                save_debug_frame_with_box(frame, (x, y, w, h), label=name)
    return enemy_visible

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

    obs_shape = env.observation_space.shape
    input_dims = (obs_shape[0] + 1, obs_shape[1], obs_shape[2]) if USE_ENEMY_CHANNEL else obs_shape
    agent = Agent(input_dims=input_dims, num_actions=env.action_space.n, use_enemy_channel=USE_ENEMY_CHANNEL)

    templates = load_templates_from(TEMPLATE_DIR) if USE_ENEMY_CHANNEL else {}
    mario_templates = load_templates_from(MARIO_TEMPLATE_DIR) if USE_ENEMY_CHANNEL else {}

    env.reset()
    env.step(action=0)

    checkpoint_name = "model_25000_iter.pt"
    checkpoint_path = os.path.join("models", TAG, checkpoint_name)
    start_episode = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        agent.load_model(checkpoint_path)
        try:
            log_path = os.path.join("models", TAG, "training_log.csv")
            with open(log_path, 'r') as f:
                reader = list(csv.DictReader(f))
                if reader:
                    last_entry = reader[-1]
                    agent.epsilon = float(last_entry.get("Epsilon", 1.0))
                    start_episode = int(last_entry.get("Episode", 0)) + 1
                    print(f"Resuming from episode {start_episode}, epsilon = {agent.epsilon}")
        except Exception as e:
            print(f"Could not restore epsilon/episode: {e}")

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

            if a in [5, 6, 7, 8] or a == 1:
                a = 3
            elif a == 2:
                a = 4

            new_state, reward, done, truncated, info = env.step(a)

            if info.get("flag_get"):
                print(f"Level completed at episode {i}")
                reward += 500

            current_x = info.get("x", 0)
            reward += (current_x - prev_x) * 0.05 if current_x > prev_x else -2
            prev_x = current_x

            if USE_ENEMY_CHANNEL and not DISPLAY:
                frame = env.render()
                if frame is not None and isinstance(frame, np.ndarray):
                    try:
                        enemy_visible = detect_enemies_and_mario(frame, templates, mario_templates)
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

    gc.collect()
    agent.save_model(os.path.join(model_path, "final_model.pt"))
    env.close()
