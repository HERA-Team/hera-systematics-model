"""Descriptive score statistics with explicit time and surrogate assumptions."""

import numpy as np
from scipy import stats


def summary(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None,
                "reason": "no finite values"}
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))}


def physical_autocorrelation(values, window_ids, max_lag=24):
    """Biased autocorrelation using pairs separated by exact native-window lags."""
    values, ids = np.asarray(values, float), np.asarray(window_ids)
    if values.ndim != 1 or values.shape != ids.shape or np.any(np.diff(ids) <= 0):
        raise ValueError("ordered score and window identity vectors required")
    if not np.isfinite(values).all():
        raise ValueError("autocorrelation needs explicitly selected finite scores")
    centered = values - values.mean()
    variance = centered @ centered
    lookup = {identity: i for i, identity in enumerate(ids)}
    acf, counts = np.full(max_lag + 1, np.nan), np.zeros(max_lag + 1, int)
    for lag in range(max_lag + 1):
        pairs = [(i, lookup[identity + lag]) for i, identity in enumerate(ids) if identity + lag in lookup]
        counts[lag] = len(pairs)
        if pairs and variance > 0:
            acf[lag] = sum(centered[i] * centered[j] for i, j in pairs) / variance
    return acf, counts


def poly_r2(time, values, degree=3):
    time, values = np.asarray(time, float), np.asarray(values, float)
    if len(time) <= degree + 1 or np.std(time) == 0 or np.var(values) == 0:
        return None
    scaled = (time - time.mean()) / time.std()
    coefficients = np.polynomial.polynomial.polyfit(scaled, values, degree)
    residual = values - np.polynomial.polynomial.polyval(scaled, coefficients)
    return float(1 - np.var(residual) / np.var(values))


def gaussianity(values):
    """Normality tests are nominal independent-sample diagnostics only."""
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("finite score vector required")
    if len(values) < 8 or np.std(values) == 0:
        return {"n": len(values), "available": False, "reason": "insufficient nonconstant scores"}
    shapiro = stats.shapiro(values)
    anderson = stats.anderson(values, dist="norm")
    critical = anderson.critical_values[np.flatnonzero(anderson.significance_level == 5)[0]]
    return {"n": len(values), "available": True, "assumption": "independent identically distributed samples",
            "skewness": float(stats.skew(values)), "excess_kurtosis": float(stats.kurtosis(values)),
            "shapiro_w": float(shapiro.statistic), "shapiro_nominal_p": float(shapiro.pvalue),
            "anderson_stat": float(anderson.statistic), "anderson_crit_5pct": float(critical)}


def qq_coordinates(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) < 2 or np.std(values) == 0:
        raise ValueError("Q-Q coordinates require finite nonconstant scores")
    standardized = (values - values.mean()) / values.std(ddof=1)
    theoretical, observed = stats.probplot(standardized, dist="norm", fit=False)
    return np.asarray(theoretical), np.asarray(observed)


def surrogate_gaussianity(values, window_ids, n_surrogates=1000, seed=0):
    """Phase-randomized diagnostic; fixed amplitudes do not define a Gaussian null.

    Uniform sampling and circular stationarity are required. A smooth trend
    can violate stationarity. Tail estimates include the observed realization.
    """
    values, ids = np.asarray(values, float), np.asarray(window_ids)
    if (len(values) < 8 or values.shape != ids.shape or not np.isfinite(values).all()
            or np.std(values) == 0 or np.any(np.diff(ids) != 1) or n_surrogates < 1):
        return {"available": False, "reason": "nonconstant scores on a continuous uniform segment required"}
    rng = np.random.default_rng(seed)
    n = len(values)
    spectrum = np.abs(np.fft.rfft(values - values.mean()))
    phases = rng.uniform(0, 2 * np.pi, (n_surrogates, len(spectrum)))
    phases[:, 0] = 0
    if n % 2 == 0:
        phases[:, -1] = rng.integers(0, 2, n_surrogates) * np.pi
    surrogates = np.fft.irfft(spectrum * np.exp(1j * phases), n=n, axis=1)
    observed = [float(stats.skew(values)), float(stats.kurtosis(values))]
    simulated = [stats.skew(surrogates, axis=1), stats.kurtosis(surrogates, axis=1)]
    tails = [(1 + np.count_nonzero(np.abs(draws) >= abs(value))) / (n_surrogates + 1)
             for value, draws in zip(observed, simulated)]
    return {"available": True, "n_surrogates": n_surrogates, "seed": seed,
            "null": "fixed Fourier amplitudes with randomized phases",
            "assumption": "uniform sampling and circular stationarity",
            "observed_skewness": observed[0], "observed_excess_kurtosis": observed[1],
            "surrogate_tail_skewness": float(tails[0]), "surrogate_tail_kurtosis": float(tails[1])}


def score_diagnostics(scores, window_ids, lst_hours, observed_fraction, degree=3):
    scores = np.asarray(scores, float)
    fractions = np.asarray(observed_fraction, float)
    if scores.ndim != 2 or scores.shape[0] != len(window_ids) or fractions.shape != (len(scores),):
        raise ValueError("score diagnostic axes disagree")
    results, arrays = [], {}
    for mode in range(scores.shape[1]):
        values = scores[:, mode]
        acf, counts = physical_autocorrelation(values, window_ids)
        arrays[f"acf_{mode}"] = acf
        arrays[f"acf_pairs_{mode}"] = counts
        distribution = gaussianity(values)
        if distribution["available"]:
            arrays[f"qq_theoretical_{mode}"], arrays[f"qq_observed_{mode}"] = qq_coordinates(values)
        association = None
        if np.std(fractions) > 0 and np.std(np.abs(values)) > 0:
            association = float(stats.spearmanr(np.abs(values), fractions).statistic)
        energy = values ** 2
        results.append({"mode": mode, "distribution": distribution, "polynomial_r2": poly_r2(lst_hours, values, degree),
            "polynomial_degree": degree, "spearman_abs_score_vs_observed_fraction": association,
            "association_unavailable_reason": "constant score magnitude or missingness" if association is None else None,
            "largest_window_energy_share": float(energy.max() / energy.sum()) if energy.sum() > 0 else None,
            "energy_unavailable_reason": "zero score energy" if energy.sum() == 0 else None})
    return results, arrays
