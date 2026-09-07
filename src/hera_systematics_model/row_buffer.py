"""Bounded batches of physically indexed visibility rows."""

import numpy as np


class RowBuffer:
    """Collect at most row_limit rows and flush in physical output-index order."""

    def __init__(self, row_limit, write):
        if type(row_limit) is not int or row_limit < 1:
            raise ValueError("a positive integer write-buffer row limit is required")
        self.row_limit = row_limit
        self.write = write
        self.parts = []
        self.rows = 0
        self.maximum_buffered_rows = 0
        self.write_calls = 0
        self.layout = None

    def append(self, indices, arrays):
        indices = np.asarray(indices)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                or np.any(indices < 0) or len(np.unique(indices)) != len(indices)):
            raise ValueError("unique nonnegative output row indices required")
        if set(arrays) != {"visdata", "flags", "nsamples"}:
            raise ValueError("all three visibility payload arrays are required")
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        if any(value.ndim != 3 or len(value) != len(indices) for value in arrays.values()):
            raise ValueError("payload rows differ from output row identities")
        layout = {key: (value.shape[1:], value.dtype) for key, value in arrays.items()}
        if len({value.shape[1:] for value in arrays.values()}) != 1:
            raise ValueError("visibility payload shapes differ")
        if self.layout is not None and layout != self.layout:
            raise ValueError("buffered payload layout changed")
        self.layout = layout
        offset = 0
        while offset < len(indices):
            end = min(len(indices), offset + self.row_limit - self.rows)
            self.parts.append((indices[offset:end].copy(),
                               {key: value[offset:end].copy() for key, value in arrays.items()}))
            self.rows += end - offset
            self.maximum_buffered_rows = max(self.maximum_buffered_rows, self.rows)
            offset = end
            if self.rows == self.row_limit:
                self.flush()

    def flush(self):
        if not self.rows:
            return
        indices = np.concatenate([part[0] for part in self.parts])
        order = np.argsort(indices, kind="stable")
        indices = indices[order]
        if len(np.unique(indices)) != len(indices):
            raise ValueError("duplicate buffered output row")
        arrays = {key: np.concatenate([part[1][key] for part in self.parts])[order]
                  for key in self.parts[0][1]}
        self.write(indices, arrays)
        self.write_calls += 1
        self.parts.clear()
        self.rows = 0
