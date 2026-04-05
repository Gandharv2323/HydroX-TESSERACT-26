"""
main_train_real.py — Entry point to train models on real pump-sensor-data.csv.

Usage:
    uv run python main_train_real.py
    uv run python main_train_real.py --csv pump-sensor-data-small.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from training.train_real_csv import train_on_real_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on real pump sensor CSV")
    parser.add_argument(
        "--csv",
        default="pump-sensor-data.csv",
        help="CSV filename (relative to this script's directory)",
    )
    parser.add_argument(
        "--out",
        default="models/real",
        help="Output directory for trained model artefacts",
    )
    args = parser.parse_args()

    csv_path = Path(__file__).parent / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    train_on_real_csv(csv_path=csv_path, out_dir=Path(__file__).parent / args.out)
