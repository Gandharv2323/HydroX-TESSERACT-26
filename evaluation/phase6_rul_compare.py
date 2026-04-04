from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reports = root / "evaluation" / "reports"
    baseline_path = reports / "phase6_rul_baseline.json"
    latest_path = reports / "phase6_rul_latest.json"

    if not baseline_path.exists() or not latest_path.exists():
        raise FileNotFoundError("Need phase6_rul_baseline.json and phase6_rul_latest.json")

    b = json.loads(baseline_path.read_text(encoding="utf-8"))
    l = json.loads(latest_path.read_text(encoding="utf-8"))

    summary = {
        "baseline": b,
        "latest": l,
        "delta_coverage": float(l["coverage"] - b["coverage"]),
        "delta_width": float(l["avg_width"] - b["avg_width"]),
        "target_band": [0.88, 0.92],
    }
    (reports / "phase6_rul_compare.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 4))
        labels = ["baseline", "latest"]
        vals = [100.0 * b["coverage"], 100.0 * l["coverage"]]
        colors = ["#a94442", "#2f7a4a"]
        plt.bar(labels, vals, color=colors)
        plt.axhspan(88, 92, alpha=0.15, color="#2f6d9f", label="target 88-92%")
        plt.ylabel("Coverage (%)")
        plt.title("RUL Conformal Coverage Improvement")
        plt.legend()
        fig.tight_layout()
        fig.savefig(reports / "phase6_rul_coverage_compare.png", dpi=150)
        plt.close(fig)

        fig = plt.figure(figsize=(7, 4))
        widths = [b["avg_width"], l["avg_width"]]
        plt.bar(labels, widths, color=["#7f8c8d", "#2f7a4a"])
        plt.axhline(220.0, color="#bb2f2f", linestyle="--", label="width target <=220h")
        plt.ylabel("Interval Width (h)")
        plt.title("RUL Interval Width Comparison")
        plt.legend()
        fig.tight_layout()
        fig.savefig(reports / "phase6_rul_width_compare.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
