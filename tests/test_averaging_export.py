import numpy as np
import pytest

from hera_systematics_model.averaging_export import memberships_from_streams, merge_memberships
from hera_systematics_model.records import WindowGrid


def inputs():
    native = 2459000. + np.arange(84) * 10 / 86400
    streams = [native[i::4] for i in range(4)]
    averaged = [stream.reshape(3, 7).mean(axis=1) for stream in streams]
    return dict(native_times=native, stream_times=streams, averaged_times=averaged,
        output_times=np.mean(averaged, axis=0), baseline_id="0_1:0_1",
        grid=WindowGrid(native[0] - 5 / 86400, 280.), samples_per_stream=7,
        metadata={"spectrum_source": {"sha256": "abc"}, "native_time_source": {"sha256": "def"}})


def test_export_uses_exact_native_ids_and_merges_reordered_rows():
    first = memberships_from_streams(**inputs())
    np.testing.assert_equal(first.native_ids, np.arange(84).reshape(3, 28))
    second = memberships_from_streams(**{**inputs(), "baseline_id": "0_2:0_2"})
    baselines = np.concatenate([second.baseline_ids, first.baseline_ids])[::-1]
    times = np.concatenate([second.centroid_jd, first.centroid_jd])[::-1]
    result = merge_memberships([first, second], baselines, times, {"sha256": "merged"})
    np.testing.assert_equal(result.baseline_ids, baselines)
    assert result.metadata["spectrum_source"] == {"sha256": "merged"}
    np.testing.assert_equal(result.native_ids[:3], first.native_ids[::-1])


def test_export_rejects_shifted_windows_approximate_ids_and_false_centroids():
    data = inputs()
    data["stream_times"][0] = data["stream_times"][0] + 1e-8
    with pytest.raises(ValueError, match="exact native"):
        memberships_from_streams(**data)
    data = inputs()
    data["averaged_times"][0] = data["averaged_times"][0] + 1e-5
    with pytest.raises(ValueError, match="consumed native"):
        memberships_from_streams(**data)
    data = inputs()
    data["output_times"] = data["output_times"] + 1e-5
    with pytest.raises(ValueError, match="interleave membership"):
        memberships_from_streams(**data)
    data = inputs()
    data["native_times"] = np.r_[data["native_times"][0] - 10 / 86400, data["native_times"]]
    with pytest.raises(ValueError, match="not aligned"):
        memberships_from_streams(**data)


def test_export_preserves_real_gaps_and_rejects_inconsistent_merge():
    data = inputs()
    data["stream_times"] = [np.r_[s[:7], s[14:]] for s in data["stream_times"]]
    data["averaged_times"] = [a[[0, 2]] for a in data["averaged_times"]]
    data["output_times"] = data["output_times"][[0, 2]]
    result = memberships_from_streams(**data)
    np.testing.assert_equal(result.window_ids, [0, 2])
    np.testing.assert_equal(result.native_ids[1], np.arange(56, 84))
    with pytest.raises(ValueError, match="duplicate"):
        merge_memberships([result, result], result.baseline_ids, result.centroid_jd, {"sha256": "merged"})
