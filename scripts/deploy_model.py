import os
import shutil
from glob import glob

MODEL_DIR = 'models'
models = glob(f'{MODEL_DIR}/model_*.joblib')

if models:
    # Get the latest model
    latest_model = max(models, key=os.path.getctime)
    target = f'{MODEL_DIR}/model_latest.joblib'
    
    # Copy to model_latest.joblib
    shutil.copy2(latest_model, target)
    print(f"✅ Deployed: {latest_model} -> {target}")
else:
    print("❌ No models found!")