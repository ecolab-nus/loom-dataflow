"""Load and validate the staged ETG JSON file."""

import json


def load_variants(json_path: str) -> list[dict]:
    """Load a staged ETG JSON file and return the list of variant dicts.

    Args:
        json_path: Path to the staged_etg_dump.json file.

    Returns:
        List of variant dicts, each conforming to the ETG schema.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON does not match the expected schema.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in '{json_path}': {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at top level, got {type(data).__name__}")

    for i, variant in enumerate(data):
        _validate_variant(variant, index=i)

    return data


def _validate_variant(v: dict, index: int) -> None:
    """Raise ValueError if a variant dict is missing required keys."""
    prefix = f"variants[{index}]"

    for key in ("variant_name", "constraint_scope", "compute_scope", "memory_scope"):
        if key not in v:
            raise ValueError(f"{prefix}: missing required key '{key}'")

    cs = v["constraint_scope"]
    if "metadata" not in cs:
        raise ValueError(f"{prefix}.constraint_scope: missing 'metadata'")
    if "hard_constraints" not in cs:
        raise ValueError(f"{prefix}.constraint_scope: missing 'hard_constraints'")

    meta = cs["metadata"]
    for key in ("symbols", "iter_num"):
        if key not in meta:
            raise ValueError(f"{prefix}.constraint_scope.metadata: missing '{key}'")

    iter_num = meta["iter_num"]
    if "seq_iter" not in iter_num:
        raise ValueError(f"{prefix}.constraint_scope.metadata.iter_num: missing 'seq_iter'")
    if "temp_iter" not in iter_num:
        raise ValueError(f"{prefix}.constraint_scope.metadata.iter_num: missing 'temp_iter'")

    for scope_key in ("compute_scope", "memory_scope"):
        scope = v[scope_key]
        if "stages" not in scope:
            raise ValueError(f"{prefix}.{scope_key}: missing 'stages'")
        for j, stage in enumerate(scope["stages"]):
            if "queues" not in stage:
                raise ValueError(f"{prefix}.{scope_key}.stages[{j}]: missing 'queues'")
            for q_name, queue in stage["queues"].items():
                if "resolved_time" not in queue:
                    raise ValueError(
                        f"{prefix}.{scope_key}.stages[{j}].queues['{q_name}']: "
                        "missing 'resolved_time'"
                    )
