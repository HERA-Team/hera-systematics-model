"""Bounded-lifetime hashing for batches that repeatedly consume shared inputs."""

from pathlib import Path
import re

from .configuration import digest_json, file_identity
from .production import write_json_exclusive


def file_stamp(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class VerifiedInputs:
    """Hash before first consumption and again before successful batch completion.

    Intermediate lookups check device, inode, size and both modification/change
    timestamps. Cached identities cannot be reused outside this one context.
    The final report is written only after every input passes a fresh hash.
    """

    def __init__(self, report_path, progress=None, expected_identities=None):
        self.report_path = Path(report_path).resolve()
        self.progress = progress
        self._expected = None
        if expected_identities is not None:
            self._expected = {}
            for item in expected_identities:
                if (not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}
                        or not isinstance(item["path"], str) or not Path(item["path"]).is_absolute()
                        or type(item["bytes"]) is not int or item["bytes"] < 0
                        or not isinstance(item["sha256"], str) or not re.match(r"[0-9a-f]{64}\Z", item["sha256"])):
                    raise ValueError("invalid expected input identity")
                path = Path(item["path"]).resolve()
                if str(path) != item["path"] or path in self._expected:
                    raise ValueError("expected input paths must be canonical and unique")
                self._expected[path] = dict(item)
            if not self._expected:
                raise ValueError("expected input inventory must not be empty")
        self._entries = {}
        self._active = False
        self._used = False

    def __enter__(self):
        if self._used:
            raise ValueError("input verification contexts cannot be reused")
        if self.report_path.exists():
            raise FileExistsError(self.report_path)
        self._used = self._active = True
        return self

    def identity(self, path):
        if not self._active:
            raise ValueError("input identity requires an active verification context")
        path = Path(path).resolve(strict=True)
        if self._expected is not None and path not in self._expected:
            raise ValueError("input absent from accepted inventory")
        stamp = file_stamp(path)
        if path not in self._entries:
            identity = file_identity(path)
            if file_stamp(path) != stamp:
                raise ValueError("input changed during initial hashing")
            if self._expected is not None and identity != self._expected[path]:
                raise ValueError("input differs from accepted inventory")
            self._entries[path] = (stamp, identity)
            if self.progress is not None:
                self.progress("initial", len(self._entries), identity)
        expected, identity = self._entries[path]
        if stamp != expected:
            raise ValueError("input changed during batch execution")
        return dict(identity)

    def __exit__(self, exception_type, exception, traceback):
        passed, failure = False, str(exception) if exception is not None else None
        try:
            if exception_type is None:
                for index, (path, (stamp, identity)) in enumerate(sorted(self._entries.items())):
                    if file_stamp(path) != stamp or file_identity(path) != identity or file_stamp(path) != stamp:
                        raise ValueError("input changed before final batch acceptance")
                    if self.progress is not None:
                        self.progress("final", index + 1, identity)
                if not self._entries:
                    raise ValueError("an empty input verification batch cannot pass")
                passed = True
        except Exception as error:
            failure = str(error)
            raise
        finally:
            self._active = False
            write_json_exclusive(self.report_path, {"schema_version": 1, "passed": passed,
                "failure": failure, "hash_checks_per_input": 2 if passed else None,
                "expected_inventory_digest": None if self._expected is None else digest_json(
                    [self._expected[path] for path in sorted(self._expected)]),
                "inputs": [dict(value[1]) for _, value in sorted(self._entries.items())],
                "intermediate_identity_fields": ["device", "inode", "bytes", "mtime_ns", "ctime_ns"]})
        return False


def input_identity(path, input_set=None):
    if input_set is None:
        return file_identity(path)
    if not isinstance(input_set, VerifiedInputs):
        raise TypeError("input_set must be an active VerifiedInputs context")
    return input_set.identity(path)
