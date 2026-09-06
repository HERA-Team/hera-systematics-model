"""RBF kernel embeddings with a separately fitted predictive decoder."""

from dataclasses import dataclass, fields

import numpy as np
from scipy.spatial.distance import cdist, pdist

from .artifacts import read_artifact, write_artifact
from .models import measured_arrays, prepare_training
from .residuals import TRANSFORMS, INVERSES
from .scoring import CandidateFailure


def median_distance_gamma(values, factor=1.):
    """Use factor times the training median squared distance as RBF width."""
    distances = pdist(values, metric="sqeuclidean")
    median = float(np.median(distances)) if len(distances) else 0.
    if not np.isfinite(median) or median <= 0:
        raise CandidateFailure("no positive median kernel distance")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("kernel width factor must be positive")
    return 1 / (factor * median)


def rbf(left, right, gamma):
    return np.exp(-gamma * cdist(left, right, metric="sqeuclidean"))


@dataclass
class KernelModel:
    input_mask: np.ndarray
    feature_mask: np.ndarray
    mean: np.ndarray
    linear_mean: np.ndarray
    training_inputs: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    kernel_mean: np.ndarray
    training_scores: np.ndarray
    dual: np.ndarray
    training_ids: np.ndarray
    metadata: dict

    @property
    def rank(self):
        return self.metadata["rank"]

    def predict(self, power, ideal, pn, valid, predictor, diagnostics=None):
        if not self.metadata.get("converged", False):
            raise CandidateFailure("kernel model did not converge")
        power, ideal, pn, valid = measured_arrays(power, ideal, pn, valid)
        predictor = np.asarray(predictor)
        if (predictor.shape != self.input_mask.shape or predictor.dtype.kind != "b"
                or power.shape[1] != len(self.mean) or np.any(self.input_mask & ~predictor)):
            raise ValueError("kernel predictor support mismatch")
        rep, params = self.metadata["representation"], self.metadata["params"]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            transformed = TRANSFORMS[rep](power[:, self.input_mask], ideal[:, self.input_mask],
                                          pn[:, self.input_mask], **params)
        if not valid[:, self.input_mask].all() or not np.isfinite(transformed).all():
            raise CandidateFailure("kernel predictor core is incomplete")
        inputs = (transformed - self.mean[self.input_mask]) / self.metadata["input_scale"]
        matrix = rbf(inputs, self.training_inputs, self.metadata["gamma"])
        centered = matrix - matrix.mean(axis=1, keepdims=True) - self.kernel_mean + self.kernel_mean.mean()
        scores = centered @ (self.eigenvectors / np.sqrt(self.eigenvalues))
        if diagnostics is not None:
            for row in range(len(power)):
                diagnostics.append({"row": row, "predictor_features": np.flatnonzero(self.input_mask).tolist(),
                    "effective_rank": self.rank, "condition_number": float(self.eigenvalues[0] / self.eigenvalues[-1]),
                    "condition_definition": "retained training kernel eigenvalue ratio"})
        decoded = rbf(scores, self.training_scores, self.metadata["decoder_gamma"]) @ self.dual
        decoded = decoded * self.metadata["target_scale"] + self.mean[self.feature_mask]
        predictions = np.broadcast_to(self.linear_mean, power.shape).copy()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            predictions[:, self.feature_mask] = INVERSES[rep](decoded, ideal[:, self.feature_mask],
                pn[:, self.feature_mask], **params) - ideal[:, self.feature_mask]
        return predictions, scores

    def save(self, path):
        arrays = {field.name: getattr(self, field.name) for field in fields(self)
                  if field.name != "metadata"}
        return write_artifact(path, "fitted-model", arrays, self.metadata)

    @classmethod
    def load(cls, path):
        arrays, metadata = read_artifact(path, "fitted-model")
        if metadata.get("method") != "kernel":
            raise ValueError("not a kernel residual model")
        model = cls(**arrays, metadata=metadata)
        nf, nt, rank = len(model.mean), len(model.training_ids), model.rank
        if (type(rank) is not int or not 1 <= rank <= min(10, nt - 2)
                or model.mean.shape != (nf,) or model.linear_mean.shape != (nf,)
                or any(mask.shape != (nf,) or mask.dtype.kind != "b" for mask in (model.input_mask, model.feature_mask))
                or np.any(model.input_mask & ~model.feature_mask)
                or model.training_ids.shape != (nt,) or model.training_ids.dtype.kind not in "iu"
                or len(np.unique(model.training_ids)) != nt
                or model.training_inputs.shape != (nt, model.input_mask.sum())
                or model.eigenvectors.shape != (nt, rank) or model.eigenvalues.shape != (rank,)
                or model.kernel_mean.shape != (nt,) or model.training_scores.shape != (nt, rank)
                or model.dual.shape != (nt, model.feature_mask.sum())):
            raise ValueError("invalid kernel state dimensions")
        finite = (model.mean, model.linear_mean[model.feature_mask], model.training_inputs,
                  model.eigenvectors, model.eigenvalues, model.kernel_mean, model.training_scores, model.dual)
        if (not all(np.isfinite(a).all() for a in finite) or np.any(model.eigenvalues <= 0)
                or metadata.get("representation") not in TRANSFORMS
                or not isinstance(metadata.get("params"), dict) or not isinstance(metadata.get("converged"), bool)
                or any(not np.isfinite(metadata[key]) or metadata[key] <= 0
                       for key in ("input_scale", "target_scale", "gamma", "decoder_gamma", "alpha"))):
            raise ValueError("invalid kernel numerical state")
        return model


def fit_kernel(power, ideal, pn, valid, rank, predictor, representation="linear", log_margin=1.,
               bandwidth=1., alpha=1., training_ids=None):
    """Fit encoder on predictors and decoder to training-complete targets."""
    from sklearn.decomposition import KernelPCA

    x, observed, linear_mean, params = prepare_training(power, ideal, pn, valid, representation, log_margin)
    predictor = np.asarray(predictor)
    if predictor.shape != (x.shape[1],) or predictor.dtype.kind != "b":
        raise ValueError("invalid kernel predictor mask")
    if bandwidth not in (.1, .3, 1., 3., 10.) or alpha not in (.1, 1., 10.):
        raise ValueError("unsupported kernel parameters")
    feature_mask = observed.all(axis=0)
    input_mask = feature_mask & predictor
    if not 1 <= rank <= min(10, len(x) - 2, input_mask.sum() - 2):
        raise CandidateFailure("kernel rank exceeds predictor support")
    mean = np.zeros(x.shape[1])
    mean[feature_mask] = x[:, feature_mask].mean(axis=0)
    inputs = x[:, input_mask] - mean[input_mask]
    targets = x[:, feature_mask] - mean[feature_mask]
    input_scale, target_scale = float(np.sqrt(np.mean(inputs ** 2))), float(np.sqrt(np.mean(targets ** 2)))
    if min(input_scale, target_scale) <= 0 or not np.isfinite([input_scale, target_scale]).all():
        raise CandidateFailure("kernel training variance is zero or nonfinite")
    inputs, targets = inputs / input_scale, targets / target_scale
    gamma = median_distance_gamma(inputs, bandwidth)
    encoder = KernelPCA(n_components=rank, kernel="rbf", gamma=gamma, eigen_solver="dense", remove_zero_eig=True)
    scores = encoder.fit_transform(inputs)
    if scores.shape[1] != rank or np.any(encoder.eigenvalues_ <= 0):
        raise CandidateFailure("kernel embedding has insufficient rank")
    decoder_gamma = median_distance_gamma(scores)
    decoder_system = rbf(scores, scores, decoder_gamma) + alpha * np.eye(len(x))
    dual = np.linalg.solve(decoder_system, targets)
    ids = np.arange(len(x)) if training_ids is None else np.asarray(training_ids)
    if ids.shape != (len(x),) or len(np.unique(ids)) != len(ids):
        raise ValueError("invalid training identities")
    return KernelModel(input_mask, feature_mask, mean, linear_mean, inputs, encoder.eigenvectors_,
        encoder.eigenvalues_, rbf(inputs, inputs, gamma).mean(axis=0), scores, dual, ids,
        {"method": "kernel", "representation": representation, "params": params, "rank": rank,
         "bandwidth": bandwidth, "alpha": alpha, "gamma": gamma, "decoder_gamma": decoder_gamma,
         "kernel_definition": "exp(-squared_distance / (bandwidth * training_median_squared_distance))",
         "encoder_retained_eigenvalue_ratio": float(encoder.eigenvalues_[0] / encoder.eigenvalues_[-1]),
         "decoder_regularized_condition": float(np.linalg.cond(decoder_system)),
         "input_scale": input_scale, "target_scale": target_scale, "converged": True})
