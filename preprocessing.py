"""Fold-fitted preprocessing; raw survey codes are never globally recoded."""

import numpy as np

IDENTIFIER_COLUMNS = {"SEQNO", "_PSU", "IDATE"}
# CDC 2015 codebook p.73: this question is asked after a reported heart attack.
# Its availability can reveal a constituent of the target, unlike general
# risk factors or other medical history. Exclude it in every comparison.
TARGET_PROXY_COLUMNS = {"HAREHAB1", "CVDINFR4", "CVDCRHD4", "_MICHD"}


class Preprocessor:
    """Impute and scale numeric features; optionally encode small discrete sets.

    Each fitted column specification is JSON-serializable. Category selection,
    imputation, scaling, and constant removal use only supplied training rows.
    The cardinality heuristic is a representation choice, not a claim about
    survey semantics. Unseen categories use the fitted reference category.
    """

    def __init__(self, imputation="mean", indicators=False, categorical=False):
        if imputation not in {"mean", "median"}:
            raise ValueError("Unknown imputation rule")
        self.config = dict(
            imputation=imputation, indicators=indicators, categorical=categorical
        )
        self.specs = []
        self.feature_names = []
        self.input_names = []

    def fit(self, x, names, indices):
        self.input_names = list(names)
        self.specs = []
        self.feature_names = ["intercept"]
        for j, name in enumerate(names):
            if name in IDENTIFIER_COLUMNS | TARGET_PROXY_COLUMNS:
                continue
            col = np.asarray(x[indices, j], dtype=np.float64)
            finite = np.isfinite(col)
            values = col[finite]
            if not values.size:
                continue
            unique, counts = np.unique(values, return_counts=True)
            if unique.size == 1 and finite.all():
                continue
            categorical = (
                self.config["categorical"]
                and unique.size <= 12
                and np.equal(unique, np.floor(unique)).all()
            )
            spec = {"index": j, "name": name, "categorical": bool(categorical)}
            if categorical:
                reference = unique[np.argmax(counts)]
                levels = unique[unique != reference].tolist()
                spec.update(reference=float(reference), levels=levels)
                self.feature_names.extend(f"{name}={v:g}" for v in levels)
            else:
                fill = float(
                    np.mean(values)
                    if self.config["imputation"] == "mean"
                    else np.median(values)
                )
                filled = np.where(finite, col, fill)
                mean = float(np.mean(filled))
                scale = float(np.std(filled))
                spec.update(fill=fill, mean=mean, scale=scale if scale > 1e-12 else 1.0)
                self.feature_names.append(name)
            missing = bool(
                self.config["indicators"] and not finite.all() and finite.any()
            )
            spec["missing_indicator"] = missing
            if missing:
                self.feature_names.append(name + "_missing")
            self.specs.append(spec)
        return self

    def transform(self, x, indices=None):
        if x.shape[1] != len(self.input_names):
            raise ValueError("Input feature count mismatch")
        if indices is None:
            indices = np.arange(x.shape[0])
        width = len(self.feature_names)
        required = len(indices) * width * np.dtype(np.float32).itemsize
        if required > 3 * 1024**3:
            raise MemoryError(f"Feature expansion would require {required} bytes")
        result = np.empty((len(indices), width), dtype=np.float32)
        for start in range(0, len(indices), 8192):
            selected = indices[start : start + 8192]
            block = np.asarray(x[selected], dtype=np.float64)
            out = result[start : start + len(selected)]
            out[:, 0] = 1
            k = 1
            for spec in self.specs:
                col = block[:, spec["index"]]
                finite = np.isfinite(col)
                if spec["categorical"]:
                    for value in spec["levels"]:
                        out[:, k] = col == value
                        k += 1
                else:
                    out[:, k] = (
                        np.where(finite, col, spec["fill"]) - spec["mean"]
                    ) / spec["scale"]
                    k += 1
                if spec["missing_indicator"]:
                    out[:, k] = ~finite
                    k += 1
            if k != width or not np.isfinite(out).all():
                raise ValueError("Preprocessing produced an invalid design matrix")
        return result

    def state(self):
        return {
            "config": self.config,
            "specs": self.specs,
            "feature_names": self.feature_names,
            "input_names": self.input_names,
        }

    @classmethod
    def from_state(cls, state):
        obj = cls(**state["config"])
        obj.specs = state["specs"]
        obj.feature_names = state["feature_names"]
        obj.input_names = state["input_names"]
        return obj
