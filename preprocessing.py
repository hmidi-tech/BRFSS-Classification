"""Fold-fitted preprocessing; raw survey codes are never globally recoded."""

import copy

import numpy as np

IDENTIFIER_COLUMNS = {"Id", "SEQNO", "_PSU", "IDATE"}
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
        self._fitted = False

    @staticmethod
    def _indices(x, indices):
        indices = np.arange(x.shape[0]) if indices is None else np.asarray(indices)
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= x.shape[0])
        ):
            raise ValueError(
                "Row indices must be a one-dimensional integer vector within bounds"
            )
        return indices

    def fit(self, x, names, indices):
        self._fitted = False
        if x.ndim != 2 or len(names) != x.shape[1] or len(set(names)) != len(names):
            raise ValueError(
                "Feature schema must have one distinct name per input column"
            )
        indices = self._indices(x, indices)
        if not len(indices):
            raise ValueError("Cannot fit preprocessing without training rows")
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
        self._fitted = True
        return self

    def transform(self, x, indices=None, *, names=None):
        if not self._fitted:
            raise ValueError("Preprocessing must be fitted before transformation")
        if (
            x.ndim != 2
            or x.shape[1] != len(self.input_names)
            or (names is not None and list(names) != self.input_names)
        ):
            raise ValueError("Input feature count mismatch")
        indices = self._indices(x, indices)
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
        return copy.deepcopy(
            {
                "config": self.config,
                "specs": self.specs,
                "feature_names": self.feature_names,
                "input_names": self.input_names,
            }
        )

    @classmethod
    def from_state(cls, state):
        state = copy.deepcopy(state)
        obj = cls(**state["config"])
        obj.specs = state["specs"]
        obj.feature_names = state["feature_names"]
        obj.input_names = state["input_names"]
        expected = ["intercept"]
        seen = set()
        if len(set(obj.input_names)) != len(obj.input_names):
            raise ValueError("Serialized schema contains duplicate names")
        for spec in obj.specs:
            j = spec["index"]
            if (
                not isinstance(j, int)
                or j < 0
                or j >= len(obj.input_names)
                or j in seen
                or spec["name"] != obj.input_names[j]
                or spec["name"] in IDENTIFIER_COLUMNS | TARGET_PROXY_COLUMNS
            ):
                raise ValueError("Serialized preprocessing schema mismatch")
            seen.add(j)
            if spec["categorical"]:
                levels = np.asarray(spec["levels"])
                if (
                    not np.isfinite(levels).all()
                    or len(set(spec["levels"])) != len(levels)
                    or spec["reference"] in spec["levels"]
                ):
                    raise ValueError("Serialized categorical schema is invalid")
                expected.extend(f"{spec['name']}={v:g}" for v in levels)
            else:
                if (
                    not np.isfinite([spec["fill"], spec["mean"], spec["scale"]]).all()
                    or spec["scale"] <= 0
                ):
                    raise ValueError("Serialized numeric schema is invalid")
                expected.append(spec["name"])
            if spec["missing_indicator"]:
                expected.append(spec["name"] + "_missing")
        if expected != obj.feature_names:
            raise ValueError("Serialized feature schema and names differ")
        obj._fitted = True
        return obj
