"""Training-only transforms and linear residual-model prediction."""

from dataclasses import dataclass, fields

import numpy as np

from .artifacts import read_artifact, write_artifact
from .residuals import TRANSFORMS, INVERSES
from .scoring import CandidateFailure, training_mean


def measured_arrays(power, ideal, pn, valid):
    if any(np.asarray(x).dtype.kind not in "fiu" for x in (power, ideal, pn)):
        raise ValueError("real numerical power and noise arrays required")
    power, ideal, pn = (np.asarray(x, dtype=float) for x in (power, ideal, pn))
    valid = np.asarray(valid)
    if (power.ndim != 2 or any(x.shape != power.shape for x in (ideal, pn, valid))
            or valid.dtype.kind != "b"):
        raise ValueError("matching row/feature arrays and boolean validity required")
    if (not all(np.isfinite(x[valid]).all() for x in (power, ideal, pn))
            or np.any(pn[valid] <= 0)):
        raise ValueError("nonfinite measured values or invalid noise")
    return power, ideal, pn, valid


def prepare_training(power, ideal, pn, valid, representation, log_margin=1.):
    power, ideal, pn, valid = measured_arrays(power, ideal, pn, valid)
    if representation not in TRANSFORMS or not valid.any():
        raise CandidateFailure("unknown representation or no training support")
    params = {}
    if representation == "log_ratio":
        if log_margin not in (1., 3., 10.):
            raise ValueError("unsupported log margin")
        minimum = min(power[valid].min(), ideal[valid].min(), 0.)
        params["floor"] = float(-minimum + log_margin * np.median(pn[valid]))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = TRANSFORMS[representation](power, ideal, pn, **params)
    observed = valid & np.isfinite(transformed)
    linear_mean = training_mean(power - ideal, valid)
    return transformed, observed, linear_mean, params


@dataclass
class LinearModel:
    components: np.ndarray
    mean: np.ndarray
    linear_mean: np.ndarray
    feature_mask: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    training_ids: np.ndarray
    metadata: dict

    @property
    def rank(self):
        return self.metadata["rank"]

    def predict(self, power, ideal, pn, valid, predictor):
        """Infer coefficients from predictor cells only; return linear residuals."""
        if not self.metadata.get("converged", False):
            raise CandidateFailure("model did not converge")
        power, ideal, pn, valid = measured_arrays(power, ideal, pn, valid)
        predictor = np.asarray(predictor)
        if predictor.shape != (power.shape[1],) or predictor.dtype.kind != "b":
            raise ValueError("a boolean predictor-feature mask is required")
        if power.shape[1] != len(self.mean):
            raise ValueError("model feature count mismatch")
        predictions = np.broadcast_to(self.linear_mean, power.shape).copy()
        scores = np.zeros((len(power), self.rank))
        if self.rank == 0:
            return predictions, scores
        representation, params = self.metadata["representation"], self.metadata["params"]
        # Slicing before transformation prevents target values entering inference.
        observed_features = np.flatnonzero(self.feature_mask & predictor)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            transformed = TRANSFORMS[representation](power[:, observed_features],
                ideal[:, observed_features], pn[:, observed_features], **params)
        basis = self.components[:self.rank]
        for row in range(len(power)):
            usable = valid[row, observed_features] & np.isfinite(transformed[row])
            indices = observed_features[usable]
            if len(indices) < self.rank + 2:
                raise CandidateFailure("insufficient observed predictor cells")
            coefficients, _, effective_rank, _ = np.linalg.lstsq(
                basis[:, indices].T, transformed[row, usable] - self.mean[indices], rcond=1e-10)
            if effective_rank < self.rank:
                raise CandidateFailure("rank-deficient predictor subspace")
            scores[row] = coefficients
            reconstructed = self.mean + coefficients @ basis
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                inverse = INVERSES[representation](reconstructed, ideal[row], pn[row], **params) - ideal[row]
            predictions[row, self.feature_mask] = inverse[self.feature_mask]
        return predictions, scores

    def save(self, path):
        arrays = {field.name: getattr(self, field.name) for field in fields(self)
                  if field.name != "metadata"}
        return write_artifact(path, "fitted-model", arrays, self.metadata)

    @classmethod
    def load(cls, path):
        arrays, metadata = read_artifact(path, "fitted-model")
        if metadata.get("method") not in ("complete", "masked"):
            raise ValueError("not a linear residual model")
        model = cls(**arrays, metadata=metadata)
        nf = len(model.mean)
        if (model.mean.ndim != 1 or model.components.ndim != 2 or model.components.shape[1] != nf
                or type(model.rank) is not int or not 0 <= model.rank <= len(model.components)
                or model.feature_mask.shape != (nf,) or model.feature_mask.dtype.kind != "b"
                or model.linear_mean.shape != (nf,)
                or model.training_ids.ndim != 1 or model.training_ids.dtype.kind not in "iu"
                or len(np.unique(model.training_ids)) != len(model.training_ids)
                or model.singular_values.ndim != 1 or len(model.singular_values) != len(model.components)
                or model.explained_variance_ratio.ndim != 1
                or len(model.explained_variance_ratio) not in (0, len(model.components))):
            raise ValueError("invalid fitted model dimensions")
        if (metadata.get("representation") not in TRANSFORMS or not isinstance(metadata.get("params"), dict)
                or not isinstance(metadata.get("converged"), bool)
                or not np.isfinite(model.mean).all() or not np.isfinite(model.components).all()
                or not np.isfinite(model.linear_mean[model.feature_mask]).all()
                or not np.isfinite(model.singular_values).all() or np.any(model.singular_values < 0)
                or not np.isfinite(model.explained_variance_ratio).all()
                or np.any(model.explained_variance_ratio < 0)
                or np.any(model.components[:, ~model.feature_mask] != 0)):
            raise ValueError("invalid fitted model numerical state")
        return model


def fit_complete(power, ideal, pn, valid, rank, representation="linear", log_margin=1., training_ids=None):
    x, observed, linear_mean, params = prepare_training(power, ideal, pn, valid, representation, log_margin)
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    mask = observed.all(axis=0)
    if not mask.any():
        raise CandidateFailure("no training-complete features")
    mean = np.zeros(x.shape[1])
    mean[mask] = x[:, mask].mean(axis=0)
    _, singular, vt = np.linalg.svd(x[:, mask] - mean[mask], full_matrices=False)
    support = int(np.count_nonzero(singular > (singular[0] * 1e-10 if len(singular) else 0)))
    if rank > min(support, len(x) - 2):
        raise CandidateFailure("rank exceeds training numerical support")
    signs = np.sign(vt[np.arange(len(vt)), np.argmax(np.abs(vt), axis=1)])
    vt *= np.where(signs == 0, 1, signs)[:, None]
    components = np.zeros((len(vt), x.shape[1]))
    components[:, mask] = vt
    energy = singular ** 2
    evr = energy / energy.sum() if energy.sum() > 0 else np.zeros_like(energy)
    ids = np.arange(len(x)) if training_ids is None else np.asarray(training_ids)
    if ids.shape != (len(x),) or len(np.unique(ids)) != len(ids):
        raise ValueError("invalid training identities")
    return LinearModel(components, mean, linear_mean, mask, singular, evr, ids,
        {"method": "complete", "representation": representation, "params": params,
         "rank": rank, "converged": True})
