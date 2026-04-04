"""validation/validate_fusion.py — Phase 5: Fusion branch-balance validation.

Uses persisted fusion training metrics to verify that IF, RF, hysteresis,
latent(PCA), and RUL all contribute measurably and no branch dominates.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pickle

DOMINANCE_LIMIT = 0.60
MIN_BRANCH_SHARE = 0.02


def main() -> None:
    print("=" * 62)
    print("  Phase 5 — Fusion Balance Ablation Test")
    print(f"  Dominance limit: {DOMINANCE_LIMIT:.0%}")
    print("=" * 62)

    pth = _ROOT / "models" / "fusion_meta.pkl"
    if not pth.exists():
        print("  ERROR: models/fusion_meta.pkl not found. Run: python main_train.py")
        sys.exit(1)

    with open(pth, "rb") as fh:
        bundle = pickle.load(fh)

    metrics = bundle.get("metrics", {})
    layout = bundle.get("feature_layout", {})
    contrib = metrics.get("contributions_l1", {})

    print(f"\n  Feature layout   : {layout}")
    print(f"  AUC / AP         : {metrics.get('roc_auc')} / {metrics.get('avg_precision')}")
    print(f"  Dominant share   : {metrics.get('dominant_share')}")
    print(f"  L1 shares        : {contrib}")

    required = ["if", "rf", "latent", "rul"]
    if int(layout.get("hysteresis", 0)) == 1:
        required.append("hysteresis")

    min_ok = all(float(contrib.get(k, 0.0)) >= MIN_BRANCH_SHARE for k in required)
    dom_ok = float(metrics.get("dominant_share", 1.0)) <= DOMINANCE_LIMIT

    print(f"\n  Branch min-share >= {MIN_BRANCH_SHARE:.0%}: {'PASS OK' if min_ok else 'FAIL X'}")
    print(f"  Dominance <= {DOMINANCE_LIMIT:.0%}       : {'PASS OK' if dom_ok else 'FAIL X'}")

    overall = min_ok and dom_ok
    print("\n" + "=" * 62)
    print(f"  OVERALL: {'PASS OK' if overall else 'FAIL X'}")
    print("=" * 62)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
