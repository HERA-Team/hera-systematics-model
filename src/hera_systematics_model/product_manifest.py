"""Hash-bound variable output inventories for immutable compute tasks."""

import json
from pathlib import Path
import re

from .production import write_json_exclusive


def _specifications(directory, products, output):
    if not isinstance(products, list) or not products:
        raise ValueError("a nonempty product inventory is required")
    seen = set()
    root = Path(directory).resolve()
    for item in products:
        if (not isinstance(item, dict) or not {"path", "kind"} <= set(item)
                or set(item) - {"path", "kind", "required_paths", "bytes", "sha256"}
                or not isinstance(item["path"], str) or not item["path"]
                or item["kind"] not in ("file", "npz", "hdf5")):
            raise ValueError("invalid manifest product specification")
        relative = Path(item["path"])
        path = (root / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root)
                or path == Path(output).resolve() or path in seen):
            raise ValueError("duplicate, escaped or self-referencing manifest product")
        if "required_paths" in item and (item["kind"] != "hdf5"
                or not isinstance(item["required_paths"], list)
                or any(not isinstance(value, str) or not value for value in item["required_paths"])):
            raise ValueError("invalid required dataset paths")
        seen.add(path)
        yield {key: item[key] for key in ("path", "kind", "required_paths") if key in item}


def create_product_manifest(directory, specifications, output):
    """Verify and bind each output before exclusively writing its inventory.

    Product paths are relative to the task directory. Inventories cannot
    recursively contain other inventories or refer outside that directory.
    """
    from .worker import verify_product

    directory = Path(directory).resolve()
    output = (directory / output).resolve()
    if not output.is_relative_to(directory):
        raise ValueError("manifest output escaped the task directory")
    products = []
    for specification in _specifications(directory, specifications, output):
        result = verify_product(directory, specification)
        products.append({**specification, "bytes": result["bytes"], "sha256": result["sha256"]})
    value = {"schema_version": 1, "products": products}
    write_json_exclusive(output, value)
    return value


def verify_product_manifest(directory, path):
    """Reopen every bound product, validate structure and require exact hashes."""
    from .worker import verify_product

    value = json.loads(Path(path).read_text())
    if (not isinstance(value, dict) or set(value) != {"schema_version", "products"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1):
        raise ValueError("unsupported product manifest schema")
    specifications = list(_specifications(directory, value["products"], path))
    checked = []
    for item, specification in zip(value["products"], specifications):
        if (type(item.get("bytes")) is not int or item["bytes"] <= 0
                or not isinstance(item.get("sha256"), str)
                or not re.match(r"[0-9a-f]{64}\Z", item["sha256"])):
            raise ValueError("manifest product lacks a valid byte count and hash")
        result = verify_product(directory, specification)
        if any(result[key] != item[key] for key in ("bytes", "sha256")):
            raise ValueError("manifest product changed")
        checked.append(result)
    return {"schema_version": 1, "products": checked}
