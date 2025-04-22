import torch
from config import get_config, MODEL_MODULES, MODEL_IDS

# === GPU CONFIGURATION (optional) ===
if torch.cuda.is_available():
    print("✅ Using CUDA:", torch.cuda.get_device_name(0))
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_num_threads(4)
else:
    print("❌ CUDA not available. Running on CPU.")
MODEL_IDS = [4, 1]
# === RUN EACH MODEL SEQUENTIALLY ===
for model_id in MODEL_IDS:
    # Dynamically import the model module
    module_name = MODEL_MODULES[model_id]
    model = __import__(module_name)

    # Generate config for the model
    config = get_config(model_id)
    config["MODEL_TAG"] = f"Model {model_id}"

    print(f"\n🚀 Launching Model {model_id}: {model.MODEL_DESCRIPTION}")
    model.run(config)
    print(f"✅ Model {model_id} finished.")
    print("=" * 50)