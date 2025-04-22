import torch

# === GPU CONFIGURATION ===
if torch.cuda.is_available():
    print("✅ Using CUDA:", torch.cuda.get_device_name(0))
    torch.cuda.set_per_process_memory_fraction(0.75, 0)  # Limit memory usage
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_num_threads(4)
else:
    print("❌ CUDA not available. Running on CPU.")

# === GLOBAL SETTINGS ===


def get_config(model_id):
    use_enemy_channel = model_id == 3 or model_id == 4
    return {
        "SHOULD_TRAIN": True,
        "DISPLAY": False,
        "NUM_OF_EPISODES": 60000,
        "CKPT_SAVE_INTERVAL": 5000,
        "ENV_NAME": "SuperMarioBros-1-1-v0",
        "MODEL_TAG": f"Model {model_id}",
        "USE_ENEMY_CHANNEL": use_enemy_channel
    }

# For threadedMain
MODEL_IDS = [1, 2, 3]
MODEL_MODULES = {
    1: "Model1",
    2: "Model2",
    3: "Model3",
    4: "Model4"
}
