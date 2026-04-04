"""Utilities for packaging and unpacking HydroX advanced model artifacts.

Usage:
  python model_bundle.py pack --output hydrox_model_bundle.zip
  python model_bundle.py unpack --bundle hydrox_model_bundle.zip
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REQUIRED_FILES = [
    "models/isolation_forest.pkl",
    "models/fault_classifier.pkl",
    "models/rul_lstm.pt",
    "models/shared_latent.pt",
    "models/fusion_meta.pkl",
    "configs/threshold.json",
    "config.json",
]

OPTIONAL_FILES = [
    "configs/score_fusion.json",
]


class BundleError(RuntimeError):
    """Raised when bundle operations cannot complete safely."""


def _missing_required(base_dir: Path) -> list[str]:
    missing: list[str] = []
    for rel in REQUIRED_FILES:
        if not (base_dir / rel).exists():
            missing.append(rel)
    return missing


def create_bundle(base_dir: Path, output_path: Path) -> Path:
    """Create a single distributable zip containing advanced model artifacts."""
    missing = _missing_required(base_dir)
    if missing:
        raise BundleError("Missing required files: " + ", ".join(missing))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in REQUIRED_FILES:
            zf.write(base_dir / rel, arcname=rel)
        for rel in OPTIONAL_FILES:
            p = base_dir / rel
            if p.exists():
                zf.write(p, arcname=rel)

    return output_path


def extract_bundle(bundle_path: Path, target_dir: Path, force: bool = False) -> list[Path]:
    """Extract bundle into workspace tree (models/configs)."""
    if not bundle_path.exists():
        raise BundleError(f"Bundle not found: {bundle_path}")

    extracted: list[Path] = []
    with zipfile.ZipFile(bundle_path, mode="r") as zf:
        names = zf.namelist()
        for rel in REQUIRED_FILES:
            if rel not in names:
                raise BundleError(f"Bundle missing required entry: {rel}")

        for name in names:
            dest = target_dir / name
            if dest.exists() and not force:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name, "r") as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted.append(dest)

    return extracted


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HydroX model bundle utility")
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack", help="Create model bundle zip")
    pack.add_argument("--base-dir", default=".", help="HydroX workspace root")
    pack.add_argument("--output", default="hydrox_model_bundle.zip", help="Output bundle path")

    unpack = sub.add_parser("unpack", help="Extract model bundle zip")
    unpack.add_argument("--base-dir", default=".", help="HydroX workspace root")
    unpack.add_argument("--bundle", default="hydrox_model_bundle.zip", help="Bundle path")
    unpack.add_argument("--force", action="store_true", help="Overwrite existing files")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()

    try:
        if args.command == "pack":
            output = Path(args.output)
            if not output.is_absolute():
                output = (base_dir / output).resolve()
            path = create_bundle(base_dir, output)
            print(f"Created bundle: {path}")
            return 0

        if args.command == "unpack":
            bundle = Path(args.bundle)
            if not bundle.is_absolute():
                bundle = (base_dir / bundle).resolve()
            extracted = extract_bundle(bundle, base_dir, force=args.force)
            print(f"Extracted {len(extracted)} files from bundle.")
            return 0

        parser.print_help()
        return 2
    except BundleError as exc:
        print(f"[model_bundle] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
