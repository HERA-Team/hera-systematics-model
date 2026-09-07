"""Validated analysis configuration and consumed software fingerprints."""

from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from .artifacts import canonical_json, sha256_file
from .prediction import candidate_grid


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass
class AnalysisConfig:
    guard: int = 12
    max_rank: int = 20
    include_kernel: bool = True
    representations: list = field(default_factory=lambda: ["linear", "noise_weighted", "signed_asinh", "log_ratio"])
    methods: list = field(default_factory=lambda: ["complete", "masked"])
    group: int | None = None
    delay: int | None = None
    group_exclusion: str | None = None
    region: str = "full"

    def __post_init__(self):
        if type(self.guard) is not int or self.guard not in (8, 12, 16):
            raise ValueError("guard must be 8, 12 or 16 native windows")
        if type(self.max_rank) is not int or not 0 <= self.max_rank <= 20:
            raise ValueError("max_rank must be between zero and 20")
        if type(self.include_kernel) is not bool:
            raise ValueError("include_kernel must be boolean")
        for value, allowed in ((self.representations, {"linear", "noise_weighted", "signed_asinh", "log_ratio"}),
                               (self.methods, {"complete", "masked"})):
            if not isinstance(value, list) or not value or len(set(value)) != len(value) or not set(value) <= allowed:
                raise ValueError("invalid or duplicate analysis choices")
        for value in (self.group, self.delay):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("slice indices must be nonnegative integers")
        if self.group is not None and self.delay is not None:
            raise ValueError("only one slice direction may be selected")
        if self.group_exclusion not in (None, "leading_linear_loading", "noise_weighted_energy"):
            raise ValueError("unsupported group exclusion measure")
        if self.group_exclusion is not None and (self.group is not None or self.delay is not None):
            raise ValueError("group exclusions require the full cylindrical plane")
        if self.region not in ("full", "horizon", "horizon_buffer"):
            raise ValueError("region must be full, horizon or horizon_buffer")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        canonical_json(data)
        return cls(**data)

    def as_dict(self):
        return asdict(self)

    def candidates(self):
        return candidate_grid(self.max_rank, self.include_kernel, self.representations, self.methods)

    def training_filter(self, samples):
        from .sensitivity import CombinedFilters, GeometricRegion, GroupExclusion
        from .views import geometry_masks

        filters = []
        if self.region != "full":
            mask = geometry_masks(samples, samples.baseline_length_m / 299792458.)[self.region]
            gs = slice(None) if self.group is None else slice(self.group, self.group + 1)
            ds = slice(None) if self.delay is None else slice(self.delay, self.delay + 1)
            filters.append(GeometricRegion(mask[gs, ds].ravel(), self.region))
        if self.group_exclusion is not None:
            counts = {6: 3, 7: 1}
            if samples.metadata["spw"] not in counts:
                raise ValueError("group exclusions are defined only for spectral windows 6 and 7")
            filters.append(GroupExclusion(samples.corrupted.shape[1:], tuple(samples.group_ids),
                                          counts[samples.metadata["spw"]], self.group_exclusion))
        return None if not filters else filters[0] if len(filters) == 1 else CombinedFilters(tuple(filters))


def file_identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def capture_runtime():
    """Hash executed package sources and record resolved distribution versions."""
    package = Path(__file__).resolve().parent
    sources = {str(path.relative_to(package)): sha256_file(path) for path in sorted(package.rglob("*.py"))}
    distributions = sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()})
    runtime = {"python": platform.python_version(), "executable": sys.executable,
               "platform": platform.platform(), "distributions": [list(item) for item in distributions],
               "package_sources": sources, "source_digest": digest_json(sources)}
    root = package.parent.parent
    if (root / ".git").exists():
        runtime["git_commit"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        runtime["git_status"] = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).splitlines()
    return runtime


def capture_imports(names):
    """Record actually imported module versions and paths, including shadowing."""
    result = {}
    for name in names:
        module = importlib.import_module(name)
        path = getattr(module, "__file__", None)
        result[name] = {"version": str(getattr(module, "__version__", "unavailable")),
                        "imported_file": file_identity(path) if path else None,
                        "file_unavailable_reason": None if path else "module has no source file"}
    return {"runtime": capture_runtime(), "imports": result}
