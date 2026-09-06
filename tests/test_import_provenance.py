import json
import sys
from types import SimpleNamespace

from hera_systematics_model.configuration import capture_imports


def test_import_fingerprint_records_executed_module_not_distribution_label(tmp_path, monkeypatch):
    source = tmp_path / "module.py"
    source.write_text("__version__ = 'source-version'\n")
    module = SimpleNamespace(__file__=str(source), __version__="source-version")
    monkeypatch.setitem(sys.modules, "spectral_fixture", module)
    evidence = capture_imports(["spectral_fixture"])
    assert evidence["imports"]["spectral_fixture"]["version"] == "source-version"
    assert evidence["imports"]["spectral_fixture"]["imported_file"]["path"] == str(source)
    first = evidence["imports"]["spectral_fixture"]["imported_file"]["sha256"]
    source.write_text("__version__ = 'changed-source'\n")
    assert capture_imports(["spectral_fixture"])["imports"]["spectral_fixture"]["imported_file"]["sha256"] != first
    json.dumps(evidence, allow_nan=False)
