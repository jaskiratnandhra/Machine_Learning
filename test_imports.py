# test_imports.py
from src.train import train_and_save

csv_path = "data/sleep.csv"
target = "Sleep Disorder"   # change if needed

print("Training on:", csv_path)
metrics = train_and_save(csv_path, target, model_out="sleep_model.joblib")
print("✅ Training successful")
print("Metrics keys:", metrics.keys())
