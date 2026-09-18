#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    staged_etg = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else repo / "build/tool/loom-opt/single_stage/staged_etg"
    )
    if not staged_etg.is_file():
        print(f"SKIP: staged_etg is unavailable at {staged_etg}")
        return 77

    fixture_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        output = Path(temp_dir) / "etg.json"
        subprocess.run(
            [
                str(staged_etg),
                "--input",
                str(fixture_dir / "per_memory_capacity_input.mlir"),
                "--hw_spec",
                str(fixture_dir / "per_memory_capacity_hw.mlir"),
                "--output",
                str(output),
            ],
            check=True,
        )
        variants = json.loads(output.read_text())

        source = (fixture_dir / "per_memory_capacity_input.mlir").read_text()
        unknown_input = temp_dir / "unknown_memory.mlir"
        unknown_input.write_text(source.replace("on @RRAM", "on @MISSING", 1))
        unknown = subprocess.run(
            [
                str(staged_etg),
                "--input",
                str(unknown_input),
                "--hw_spec",
                str(fixture_dir / "per_memory_capacity_hw.mlir"),
                "--output",
                str(output),
            ],
            text=True,
            capture_output=True,
        )
        assert unknown.returncode != 0
        assert "@mem_MISSING" in unknown.stderr

        buffered_input = temp_dir / "explicit_buffer_count.mlir"
        buffered_input.write_text(
            source.replace(
                "on @RRAM :", "on @RRAM {buffer_count = 2} :", 1
            )
        )
        buffered = subprocess.run(
            [
                str(staged_etg),
                "--input",
                str(buffered_input),
                "--hw_spec",
                str(fixture_dir / "per_memory_capacity_hw.mlir"),
                "--output",
                str(output),
            ],
            text=True,
            capture_output=True,
        )
        assert buffered.returncode != 0
        assert "requires buffer_count = 1" in buffered.stderr

    assert len(variants) == 1
    metadata = variants[0]["constraint_scope"]["metadata"]
    assert "L1_footprint" not in metadata
    assert "datatype" not in metadata
    footprints = metadata["memory_footprints"]
    assert [record["memory"] for record in footprints] == ["RRAM", "SRAM"]
    assert [record["capacity_bytes"] for record in footprints] == [4096, 8192]

    rram, sram = footprints
    assert rram["load_bytes"] == [
        {"Mul": [{"Mul": [{"Sym": "K"}, {"Const": 32}]}, {"Const": 2}]}
    ]
    assert sram["load_bytes"] == [
        {"Mul": [{"Mul": [{"Const": 32}, {"Sym": "K"}]}, {"Const": 2}]}
    ]
    assert sram["compute_bytes"] == [
        {"Mul": [{"Mul": [{"Const": 32}, {"Const": 32}]}, {"Const": 4}]},
    ]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
