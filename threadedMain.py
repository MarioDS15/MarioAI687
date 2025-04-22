import threading
import importlib
from config import MODEL_IDS, MODEL_MODULES, get_config

def run_model(model_id):
    model = importlib.import_module(MODEL_MODULES[model_id])
    config = get_config(model_id)
    print(f"Launching Model {model_id}: {model.MODEL_DESCRIPTION}")
    model.run(config)

threads = []
for model_id in MODEL_IDS[:-1]:
    t = threading.Thread(target=run_model, args=(model_id,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
