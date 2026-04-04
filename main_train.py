"""
main_train.py — Entry point: train all models.

Run from pump_twin/ directory:
    python main_train.py

This will:
  1. Generate 300 synthetic samples per fault class
  2. Train Isolation Forest (anomaly detection)
  3. Train 5-class Random Forest (fault classification)
  4. Train LSTM (RUL prediction)
  5. Save all models to models/
  6. Print evaluation metrics
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from training.train_all import main

if __name__ == "__main__":
    main()
