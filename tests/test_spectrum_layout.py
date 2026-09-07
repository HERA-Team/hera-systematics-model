import h5py
import numpy as np
import pytest

from hera_systematics_model.spectrum_layout import inspect_spectrum, require_compatible


def make_spectrum(path, baseline=106134, start=10., rows=3):
    with h5py.File(path, "w") as handle:
        handle.attrs["pspec_type"] = "PSpecContainer"
        header = handle.create_group("header")
        header.attrs["hera_pspec.version"] = "0.4.3"
        header.attrs["frf_losses"] = np.arange(14.) + baseline
        group = handle.create_group("stokespol/interleave_averaged")
        group.attrs.update(Nbls=1, Nblpairs=1, Nbltpairs=rows, Ntimes=rows * 2,
                           Ntpairs=rows, Nspws=14, Npols=2, folded=False, history="source history")
        for name in ("vis_units", "norm_units", "cosmo", "taper", "norm"):
            group.attrs[name] = name
        group.attrs["spw_array"] = np.arange(14)
        group.attrs["polpair_array"] = [2121, 2222]
        group.attrs["scalar_array"] = np.ones((14, 2))
        group.attrs["channel_width"] = np.ones(42)
        group.attrs["telescope_location"] = [1., 2., 3.]
        group.attrs["dly_array"] = np.tile([-1., 0., 1.], 14)
        group.attrs["freq_array"] = np.arange(42.)
        group["spw_dly_array"] = np.repeat(np.arange(14), 3)
        group["spw_freq_array"] = np.repeat(np.arange(14), 3)
        for name in ("OmegaP", "OmegaPP"):
            group[name] = np.ones((4, 2))
        group["bl_array"] = [baseline]
        group["bl_vecs"] = [[1., float(baseline), 0.]]
        group["blpair_array"] = np.repeat(baseline * 1000001, rows)
        for axis in ("time", "lst"):
            for component, offset in (("1", 0.), ("2", .1), ("avg", .05)):
                group[f"{axis}_{component}_array"] = np.arange(rows) + start + offset
        for name in ("label_1_array", "label_2_array"):
            group[name] = np.zeros((14, rows, 2), dtype=int)
        for spw in range(14):
            for prefix in ("data_spw", "stats_P_N_", "stats_P_SN_"):
                values = np.arange(rows * 6).reshape(rows, 3, 2).astype(complex)
                values[0, 0, 0] = np.nan
                group[f"{prefix}{spw}"] = values + baseline
            group[f"wgt_spw{spw}"] = np.ones((rows, 3, 2, 2))
            for prefix in ("nsample_spw", "integration_spw"):
                group[f"{prefix}{spw}"] = np.ones((rows, 2))
    return path


def test_layout_allows_different_baselines_and_times_but_requires_exact_conventions(tmp_path):
    a, b = make_spectrum(tmp_path / "a.h5"), make_spectrum(tmp_path / "b.h5", baseline=100191, start=11., rows=4)
    with h5py.File(a) as left, h5py.File(b, "r+") as right:
        x, y = left["stokespol/interleave_averaged"], right["stokespol/interleave_averaged"]
        assert inspect_spectrum(x)["counts"]["Nbltpairs"] == 3
        assert inspect_spectrum(y)["counts"]["Nbltpairs"] == 4
        require_compatible(x, y)
        for attribute in ("vis_units", "cosmo", "scalar_array", "dly_array"):
            original = y.attrs[attribute]
            y.attrs[attribute] = "changed" if isinstance(original, str) else original + 1
            with pytest.raises(ValueError, match="metadata mismatch"):
                require_compatible(x, y)
            y.attrs[attribute] = original


@pytest.mark.parametrize("mutation,reason", [
    (lambda g: g.attrs.__setitem__("spw_array", np.arange(13)), "spectral windows"),
    (lambda g: g.__delitem__("data_spw13"), "datasets are absent"),
    (lambda g: g.__setitem__("unrecognized", [1]), "unrecognized"),
    (lambda g: g["time_1_array"].__setitem__(1, np.nan), "row coordinates"),
    (lambda g: g["bl_array"].__setitem__(0, 100191), "matching geometry"),
    (lambda g: g.attrs.__setitem__("Ntimes", 99), "dimension attribute"),
    (lambda g: g.attrs.__setitem__("scalar_array", np.ones((14, 1))), "normalization dimensions"),
])
def test_invalid_spectral_structures_are_rejected(tmp_path, mutation, reason):
    path = make_spectrum(tmp_path / "input.h5")
    with h5py.File(path, "r+") as handle:
        group = handle["stokespol/interleave_averaged"]
        mutation(group)
        with pytest.raises(ValueError, match=reason):
            inspect_spectrum(group)


def test_duplicate_identity_is_rejected_even_with_consistent_shapes(tmp_path):
    path = make_spectrum(tmp_path / "input.h5")
    with h5py.File(path, "r+") as handle:
        group = handle["stokespol/interleave_averaged"]
        for name in ("time_1_array", "time_2_array"):
            group[name][1] = group[name][0]
        with pytest.raises(ValueError, match="duplicate physical"):
            inspect_spectrum(group)
