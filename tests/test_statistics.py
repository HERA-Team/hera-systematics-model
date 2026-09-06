import numpy as np

from hera_systematics_model.artifacts import canonical_json
from hera_systematics_model.statistics import physical_autocorrelation, score_diagnostics, surrogate_gaussianity


def test_autocorrelation_does_not_bridge_missing_native_windows():
    acf, counts = physical_autocorrelation([1, 2, 4, 5], [0, 1, 4, 5], max_lag=3)
    assert counts.tolist() == [4, 2, 0, 1]
    assert np.isnan(acf[2])


def test_surrogate_tails_have_finite_simulation_correction():
    values = np.zeros(80)
    values[7] = 100
    result = surrogate_gaussianity(values, np.arange(80), n_surrogates=19)
    assert .05 <= result["surrogate_tail_kurtosis"] <= 1
    assert .05 <= result["surrogate_tail_skewness"] <= 1
    assert "fixed Fourier" in result["null"]
    assert not surrogate_gaussianity(values, np.arange(80) * 2)["available"]


def test_constant_score_diagnostics_use_defined_json_unavailability():
    results, arrays = score_diagnostics(np.zeros((40, 1)), np.arange(40), np.arange(40), np.ones(40))
    assert not results[0]["distribution"]["available"]
    assert results[0]["largest_window_energy_share"] is None
    canonical_json(results)
