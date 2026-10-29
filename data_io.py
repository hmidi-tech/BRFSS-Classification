"""Validated CSV loading with bounded parsing memory and optional disk caches."""

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, content):
    """Write a generated JSON output; callers must protect historical paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def current_commit():
    """Return local source provenance, or None outside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_features(path, cache_dir):
    """Read numeric CSV, validating row lengths and integral, unique IDs."""
    path = Path(path)
    fingerprint = sha256(path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # A new cache version leaves historical caches untouched. Append extensions
    # rather than replacing a suffix in a source filename containing dots.
    stem = cache_dir / (path.stem + "_" + fingerprint[:16] + "_v2")
    array_path = Path(str(stem) + ".npy")
    manifest_path = Path(str(stem) + ".json")
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if (
            not header
            or len(header) < 2
            or header[0] != "Id"
            or len(set(header)) != len(header)
        ):
            raise ValueError(f"Invalid or duplicated header in {path}")
        if array_path.exists() and manifest_path.exists():
            meta = json.loads(manifest_path.read_text())
            if meta["sha256"] != fingerprint or meta["header"] != header:
                raise ValueError("Cached data manifest mismatch")
            if sha256(array_path) != meta.get("array_sha256"):
                raise ValueError(
                    "Cached array checksum mismatch; use a fresh cache directory"
                )
            raw = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if list(raw.shape) != meta["shape"] or raw.dtype != np.float64:
                raise ValueError("Cached array shape mismatch")
        else:
            if array_path.exists() or manifest_path.exists():
                raise ValueError("Incomplete cache; use a fresh cache directory")
            nrows = sum(1 for _ in reader)
            if nrows == 0:
                raise ValueError(f"No observations in {path}")
            with tempfile.TemporaryDirectory(
                prefix=".parse-", dir=cache_dir
            ) as temporary:
                temporary_array = Path(temporary) / "data.npy"
                raw = np.lib.format.open_memmap(
                    temporary_array,
                    mode="w+",
                    dtype=np.float64,
                    shape=(nrows, len(header)),
                )
                handle.seek(0)
                next(reader)
                start = 0
                batch = []
                for line, row in enumerate(reader, start=2):
                    if len(row) != len(header):
                        raise ValueError(f"{path}:{line}: inconsistent row length")
                    batch.append([value.strip() or "nan" for value in row])
                    if len(batch) == 4096:
                        raw[start : start + len(batch)] = np.asarray(batch, dtype=float)
                        start += len(batch)
                        batch.clear()
                if batch:
                    raw[start : start + len(batch)] = np.asarray(batch, dtype=float)
                    start += len(batch)
                if start != nrows or sha256(path) != fingerprint:
                    raise ValueError("Source changed during parsing")
                raw.flush()
                shape = list(raw.shape)
                del raw
                temporary_manifest = Path(temporary) / "manifest.json"
                write_json(
                    temporary_manifest,
                    {
                        "version": 2,
                        "sha256": fingerprint,
                        "header": header,
                        "shape": shape,
                        "array_sha256": sha256(temporary_array),
                    },
                )
                os.replace(temporary_array, array_path)
                os.replace(temporary_manifest, manifest_path)
            raw = np.load(array_path, mmap_mode="r", allow_pickle=False)
    ids_float = raw[:, 0]
    if (
        not np.isfinite(ids_float).all()
        or not np.equal(ids_float, np.floor(ids_float)).all()
        or np.any(np.abs(ids_float) >= 2**53)
    ):
        raise ValueError("IDs must be finite integers with magnitude below 2**53")
    ids = ids_float.astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise ValueError("Duplicate IDs")
    return raw[:, 1:], ids, header[1:], fingerprint


load_features = _read_features


def load_data(data_dir, cache_dir="artifacts/cache", include_test=True):
    """Load labeled train data and optionally official test data, with ID checks."""
    directory = Path(data_dir)
    x, ids, names, train_hash = _read_features(directory / "x_train.csv", cache_dir)
    with (directory / "y_train.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    if (
        len(rows) < 2
        or rows[0] != ["Id", "_MICHD"]
        or any(len(row) != 2 for row in rows[1:])
    ):
        raise ValueError("Invalid label schema")
    labels = np.asarray(rows[1:], dtype=np.int64)
    if not np.array_equal(ids, labels[:, 0]):
        raise ValueError("Feature/label IDs or row order differ")
    y = labels[:, 1]
    if not np.isin(y, [-1, 1]).all():
        raise ValueError("Dataset labels must be -1/+1")
    result = {
        "x_train": x,
        "y": y,
        "train_ids": ids,
        "names": names,
        "fingerprints": {
            "x_train.csv": train_hash,
            "y_train.csv": sha256(directory / "y_train.csv"),
        },
    }
    if include_test:
        xt, test_ids, test_names, test_hash = _read_features(
            directory / "x_test.csv", cache_dir
        )
        if names != test_names:
            raise ValueError("Train/test feature names or column order differ")
        if np.intersect1d(ids, test_ids).size:
            raise ValueError("Train/test IDs overlap")
        result.update(x_test=xt, test_ids=test_ids)
        result["fingerprints"]["x_test.csv"] = test_hash
    return result
