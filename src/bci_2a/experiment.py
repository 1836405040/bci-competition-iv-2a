"""Command-line entry point for subject-wise BCI 2a experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .data import download_subject, load_subject
from .evaluation import CVResult, evaluate_classical, evaluate_eegnet, seed_everything
from .preprocessing import make_csp_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--models", nargs="+", choices=["csp-lda", "csp-svm", "eegnet"], default=["csp-lda", "csp-svm", "eegnet"])
    parser.add_argument("--eegnet-epochs", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="download missing AxxT.gdf files")
    return parser.parse_args()


def run(args: argparse.Namespace) -> pd.DataFrame:
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for subject in args.subjects:
        path = download_subject(subject, args.data_root) if args.download else args.data_root / f"A{subject:02d}T.gdf"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Use --download or place the GDF file there.")
        X, y, sfreq, channels = load_subject(path)
        metadata = {"subject": subject, "sfreq": sfreq, "channels": channels, "n_trials": len(y)}
        (args.output / f"subject_{subject:02d}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for model in args.models:
            if model == "eegnet":
                result = evaluate_eegnet(X, y, subject, folds=args.folds, seed=args.seed, epochs=args.eegnet_epochs)
            else:
                classifier = model.split("-")[-1]
                result = evaluate_classical(X, y, make_csp_pipeline(classifier, seed=args.seed), subject, model, args.folds, args.seed)
            rows.append(asdict(result))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "subject_results.csv", index=False)
    if not frame.empty:
        frame.groupby("model", as_index=False).agg(accuracy_mean=("accuracy_mean", "mean"), accuracy_std_across_subjects=("accuracy_mean", "std"), balanced_accuracy_mean=("balanced_accuracy_mean", "mean")).to_csv(args.output / "summary.csv", index=False)
    return frame


def main() -> None:
    args = parse_args()
    frame = run(args)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
