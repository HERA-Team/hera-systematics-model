"""Constant-model state and hash-linked collections of predictive models."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import read_artifact, sha256_file, write_artifact
from .models import LinearModel, measured_arrays
from .scoring import training_mean


@dataclass
class ConstantModel:
    linear_mean: np.ndarray
    training_ids: np.ndarray
    metadata: dict

    @property
    def rank(self):
        return 0

    @property
    def feature_mask(self):
        return np.zeros(len(self.linear_mean), bool)

    def predict(self, power, ideal, pn, valid, predictor):
        power, ideal, pn, valid = measured_arrays(power, ideal, pn, valid)
        if power.shape[1] != len(self.linear_mean):
            raise ValueError("constant model feature mismatch")
        values = np.zeros_like(self.linear_mean) if self.metadata["method"] == "zero" else self.linear_mean
        return np.broadcast_to(values, power.shape).copy(), np.zeros((len(power), 0))

    def save(self, path):
        return write_artifact(path, "fitted-model", {"linear_mean": self.linear_mean,
            "training_ids": self.training_ids}, self.metadata)

    @classmethod
    def fit(cls, arrays, training_ids, method):
        if method not in ("zero", "mean"):
            raise ValueError("unknown constant model")
        power, ideal, pn, valid = measured_arrays(*arrays)
        return cls(training_mean(power - ideal, valid), np.asarray(training_ids),
                   {"method": method, "rank": 0, "representation": "linear", "converged": True})


def load_model(path):
    arrays, metadata = read_artifact(path, "fitted-model")
    method = metadata.get("method")
    if method in ("zero", "mean"):
        model = ConstantModel(**arrays, metadata=metadata)
        if model.linear_mean.ndim != 1 or model.training_ids.ndim != 1 or metadata.get("rank") != 0:
            raise ValueError("invalid constant model state")
        return model
    if method in ("complete", "masked"):
        return LinearModel.load(path)
    if method == "kernel":
        from .kernel import KernelModel
        return KernelModel.load(path)
    raise ValueError("unsupported model method")


def save_model_collection(path, models, identity=None):
    """Write separate model artifacts and return relative paths and both hashes."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    references = {}
    for label, entries in models.items():
        references[str(label)] = []
        for index, model in enumerate(entries):
            destination = path / f"model-{len(references) - 1}-{index}.npz"
            if identity is not None:
                model.metadata["identity"] = identity
            model.save(destination)
            references[str(label)].append({"path": f"{path.name}/{destination.name}",
                "sha256": sha256_file(destination), "metadata_sha256": sha256_file(destination.with_suffix(".json"))})
    return references


def load_model_collection(parent, references, identity=None):
    parent = Path(parent).resolve()
    result = {}
    for label, entries in references.items():
        result[label] = []
        for entry in entries:
            path = (parent / entry["path"]).resolve()
            if not path.is_relative_to(parent):
                raise ValueError("model reference escapes artifact directory")
            if (sha256_file(path) != entry["sha256"]
                    or sha256_file(path.with_suffix(".json")) != entry["metadata_sha256"]):
                raise ValueError("model reference hash mismatch")
            model = load_model(path)
            if identity is not None and model.metadata.get("identity") != identity:
                raise ValueError("model physical identity mismatch")
            result[label].append(model)
    return result


def predict_samples(model, samples, predictor, group=None, delay=None):
    """Require exact fitted physical feature identities before array inference."""
    from .views import analysis_view

    samples.validate()
    arrays, _, identity = analysis_view(samples, group, delay)
    if model.metadata.get("identity") != identity:
        raise ValueError("prediction sample identities differ from fitted features")
    return model.predict(*arrays, predictor=predictor)
