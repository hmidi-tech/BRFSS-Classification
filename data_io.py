"""Validated CSV loading with bounded parsing memory and optional disk caches."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_features(path, cache_dir):
    """Read numeric CSV, validating row lengths and integral, unique IDs."""
    path = Path(path)
    fingerprint = sha256(path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_dir / (path.stem + "_" + fingerprint[:16])
    array_path = stem.with_suffix(".npy")
    manifest_path = stem.with_suffix(".json")
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[0] != "Id" or len(set(header)) != len(header):
            raise ValueError(f"Invalid or duplicated header in {path}")
        if array_path.exists() and manifest_path.exists():
            meta = json.loads(manifest_path.read_text())
            if meta["sha256"] != fingerprint or meta["header"] != header:
                raise ValueError("Cached data manifest mismatch")
            raw = np.load(array_path, mmap_mode="r")
            if list(raw.shape) != meta["shape"]:
                raise ValueError("Cached array shape mismatch")
        else:
            nrows = sum(1 for _ in reader)
            raw = np.lib.format.open_memmap(
                array_path, mode="w+", dtype=np.float64, shape=(nrows, len(header))
            )
            handle.seek(0)
            next(reader)
            start = 0
            batch = []
            for line, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(f"{path}:{line}: inconsistent row length")
                batch.append([value if value else "nan" for value in row])
                if len(batch) == 4096:
                    raw[start : start + len(batch)] = np.asarray(batch, dtype=float)
                    start += len(batch)
                    batch.clear()
            if batch:
                raw[start : start + len(batch)] = np.asarray(batch, dtype=float)
                start += len(batch)
            if start != nrows:
                raise ValueError("Row count changed during parsing")
            raw.flush()
            manifest_path.write_text(
                json.dumps(
                    {"sha256": fingerprint, "header": header, "shape": list(raw.shape)},
                    indent=2,
                )
                + "\n"
            )
            del raw
            raw = np.load(array_path, mmap_mode="r")
    ids_float = raw[:, 0]
    if (
        not np.isfinite(ids_float).all()
        or not np.equal(ids_float, np.floor(ids_float)).all()
    ):
        raise ValueError("IDs must be finite integers")
    ids = ids_float.astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise ValueError("Duplicate IDs")
    return raw[:, 1:], ids, header[1:], fingerprint


def load_data(data_dir, cache_dir="artifacts/cache", include_test=True):
    """Load labeled train data and optionally official test data, with ID checks."""
    directory = Path(data_dir)
    x, ids, names, train_hash = _read_features(directory / "x_train.csv", cache_dir)
    with (directory / "y_train.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    if rows[0] != ["Id", "_MICHD"] or any(len(row) != 2 for row in rows[1:]):
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
