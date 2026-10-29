"""Refit a frozen selected configuration on all labeled training observations."""

import argparse
import json
import os
import platform
import resource
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from data_io import current_commit, load_data, sha256, write_json
from models import fit_model
from preprocessing import Preprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--config", default="config/selected_model.json")
    parser.add_argument("--model", default="artifacts/final_model.npz")
    args = parser.parse_args()
    model_path = Path(args.model)
    metadata_path = model_path.with_suffix(".json")
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError(
            "Preserve existing final weights; supply a different --model path to refit."
        )
    selected = json.loads(Path(args.config).read_text())
    d = load_data(args.data_dir, include_test=False)
    if d["fingerprints"] != selected["dataset_fingerprints"]:
        raise ValueError("Training data fingerprint differs from model selection")
    pre = Preprocessor(**selected["model"]["preprocessing"]).fit(
        d["x_train"], d["names"], np.arange(len(d["y"]))
    )
    x = pre.transform(d["x_train"])
    w, info = fit_model(x, (d["y"] == 1).astype(float), selected["model"])
    if not np.isfinite(w).all():
        raise FloatingPointError("Final training produced invalid weights")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(model_path, weights=w)
    metadata = {
        "trained_at_zurich": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
        "preprocessor": pre.state(),
        "selection": selected,
        "training_rows": len(d["y"]),
        "git_commit": current_commit(),
        "fit": info,
        "weights_sha256": sha256(model_path),
        "config_sha256": sha256(args.config),
        "training_fingerprints": d["fingerprints"],
        "sources": {
            p: sha256(p)
            for p in [
                "data_io.py",
                "preprocessing.py",
                "models.py",
                "evaluation.py",
                "train_final.py",
            ]
        },
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "model": str(model_path),
                "metadata": str(metadata_path),
                "rows": len(d["y"]),
                "features": x.shape[1],
                "fit": {k: v for k, v in info.items() if k != "trace"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
