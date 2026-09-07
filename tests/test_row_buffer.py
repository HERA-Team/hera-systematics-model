import numpy as np
import pytest

from hera_systematics_model.row_buffer import RowBuffer


def payload(indices):
    values = np.asarray(indices)[:, None, None] * np.ones((1, 3, 2))
    return {"visdata": values.astype(complex), "flags": values == 2,
            "nsamples": (values != 2).astype(np.float32)}


def test_bounded_writes_preserve_reordered_rows_and_owned_arrays():
    writes = []
    buffer = RowBuffer(4, lambda indices, arrays: writes.append((indices, arrays)))
    arrays = payload([3, 0])
    buffer.append(np.array([3, 0]), arrays)
    arrays["visdata"][:] = -99
    buffer.append(np.array([2, 1, 4, 6, 5]), payload([2, 1, 4, 6, 5]))
    assert buffer.rows == 3 and buffer.maximum_buffered_rows == 4
    buffer.flush()
    buffer.flush()
    assert buffer.write_calls == 2
    for indices, actual in writes:
        np.testing.assert_equal(indices, np.sort(indices))
        for key, truth in payload(indices).items():
            np.testing.assert_equal(actual[key], truth)
    np.testing.assert_equal(np.concatenate([x[0] for x in writes]), np.arange(7))


def test_large_input_is_split_and_failed_flush_is_not_counted():
    sizes = []
    buffer = RowBuffer(3, lambda indices, arrays: sizes.append(len(indices)))
    buffer.append(np.arange(10), payload(np.arange(10)))
    buffer.flush()
    assert sizes == [3, 3, 3, 1] and buffer.maximum_buffered_rows == 3
    def fail(indices, arrays):
        raise OSError("write failure")
    buffer = RowBuffer(2, fail)
    with pytest.raises(OSError):
        buffer.append(np.arange(2), payload(np.arange(2)))
    assert buffer.write_calls == 0 and buffer.rows == 2


def test_duplicate_or_changed_payload_fails():
    buffer = RowBuffer(4, lambda *args: None)
    buffer.append(np.array([0]), payload([0]))
    with pytest.raises(ValueError, match="layout changed"):
        buffer.append(np.array([1]), {key: value[:, :1] for key, value in payload([1]).items()})
    buffer.append(np.array([0]), payload([0]))
    with pytest.raises(ValueError, match="duplicate"):
        buffer.flush()
    for limit in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            RowBuffer(limit, lambda *args: None)
