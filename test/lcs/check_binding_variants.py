#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path


def walk_placed(node):
    if isinstance(node, dict):
        if "PlacedFunc" in node:
            yield node["PlacedFunc"]
        for value in node.values():
            yield from walk_placed(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk_placed(value)


def run(
    pipeline,
    source,
    hardware,
    *,
    explicit_memory=True,
    enumerate_bindings=True,
):
    return pipeline.run_exploration_pipeline(
        source,
        str(hardware),
        True,
        False,
        True,
        True,
        explicit_memory,
        enumerate_bindings,
    )


def rewrite_mover_destination(hardware: str, module: str) -> str:
    start = hardware.index(f"  module @{module}")
    end = hardware.index("\n  module @", start + 1)
    mover = hardware[start:end].replace("@mem_SRAM", "@mem_UNROUTED").replace(
        "@mem_RRAM", "@mem_UNROUTED"
    )
    return hardware[:start] + mover + hardware[end:]


def main() -> int:
    build_lib = Path(sys.argv[1])
    sys.path.insert(0, str(build_lib))
    import _loom_pipeline

    fixtures = Path(__file__).resolve().parent
    source = (fixtures / "binding_variants_input.mlir").read_text()
    # This fixture models memory-binding output, whose L1 allocation is a
    # placeholder to be resolved by processor selection.
    source = source.replace(
        " on @SRAM :", " on @SRAM {loom.inferred_residency} :"
    ).replace(", 1>", ">")
    hardware = fixtures / "binding_variants_hw.mlir"
    hardware_text = hardware.read_text()

    # Two independent matmul sites and one add produce a per-site Cartesian product.
    error, explored, etg = run(_loom_pipeline, source, hardware)
    assert not error, error
    variants = json.loads(etg)
    assert len(variants) == 8
    combinations = []
    for variant in variants:
        placed = list(walk_placed(variant))
        matrix = [item for item in placed if item["func"]["name"] == "matmul"]
        vector = [item for item in placed if item["func"]["name"] == "add"]
        assert len(matrix) == 2 and len(vector) == 1
        combinations.append((
            *sorted(item["target"]["array"] for item in matrix),
            vector[0]["target"]["array"],
        ))
    assert Counter(combinations) == Counter({
        ("matrix_rram", "matrix_rram", "vector_rram"): 1,
        ("matrix_rram", "matrix_rram", "vector_sram"): 1,
        ("matrix_rram", "matrix_sram", "vector_rram"): 2,
        ("matrix_rram", "matrix_sram", "vector_sram"): 2,
        ("matrix_sram", "matrix_sram", "vector_rram"): 1,
        ("matrix_sram", "matrix_sram", "vector_sram"): 1,
    })
    assert "binding_combination_count\\22:8" in explored

    # A fused scalar chain is bound per primitive. Only same-residency
    # sub/exp pairs are legal; the cross-memory pairs would require an
    # implicit transfer even though movers exist in the architecture.
    compound = (fixtures / "binding_compound_input.mlir").read_text()
    error, compound_ir, compound_etg = run(
        _loom_pipeline, compound, hardware
    )
    assert not error, error
    compound_variants = json.loads(compound_etg)
    assert len(compound_variants) == 2
    for variant in compound_variants:
        placed = [
            item
            for item in walk_placed(variant)
            if item["func"]["name"] in {"sub", "exp"}
        ]
        assert len(placed) == 2
        assert len({item["target"]["array"] for item in placed}) == 1
        sub, exp = (
            next(item for item in placed if item["func"]["name"] == name)
            for name in ("sub", "exp")
        )
        assert sub["func"]["write"] == ""
        assert exp["func"]["read"] == ""
        assert len(variant["kernel_block"]["compute_scope"]["stages"]) == 2
    assert "internal producer-consumer edge requires an implicit transfer" in compound_ir
    assert "arith.subf" in compound_ir and "math.exp" in compound_ir

    # A scalar result that is both consumed in the body and yielded constrains
    # both edges; yielding it must not hide the internal sub -> exp edge.
    yielded_internal = (
        fixtures / "binding_yielded_internal_input.mlir"
    ).read_text()
    error, yielded_ir, yielded_etg = run(
        _loom_pipeline, yielded_internal, hardware
    )
    assert not error, error
    yielded_variants = json.loads(yielded_etg)
    assert len(yielded_variants) == 2
    for variant in yielded_variants:
        placed = [
            item for item in walk_placed(variant)
            if item["func"]["name"] in {"sub", "exp"}
        ]
        assert len(placed) == 2
        assert len({item["target"]["array"] for item in placed}) == 1
    assert "internal producer-consumer edge requires an implicit transfer" in yielded_ir

    error, fixed_compound_ir, fixed_compound_etg = run(
        _loom_pipeline, compound, hardware, enumerate_bindings=False
    )
    assert not error, error
    assert "__binding_" not in fixed_compound_ir
    assert "__movers_" not in fixed_compound_ir
    assert "loom.binding_manifest" not in fixed_compound_ir
    fixed_compute = [
        item
        for item in walk_placed(json.loads(fixed_compound_etg))
        if item["func"]["name"] in {"sub", "exp"}
    ]
    assert {item["target"]["array"] for item in fixed_compute} == {
        "vector_rram"
    }

    pinned = compound.replace(
        "%sub = arith.subf %lhs, %rhs : f16",
        '%sub = arith.subf %lhs, %rhs {loom.processor_array = "vector_sram", loom.processor_function = "sub"} : f16',
    ).replace(
        "%exp = math.exp %sub : f16",
        '%exp = math.exp %sub {loom.processor_array = "vector_sram", loom.processor_function = "exp"} : f16',
    )
    error, _, pinned_etg = run(
        _loom_pipeline, pinned, hardware, enumerate_bindings=False
    )
    assert not error, error
    assert {
        item["target"]["array"]
        for item in walk_placed(json.loads(pinned_etg))
        if item["func"]["name"] in {"sub", "exp"}
    } == {"vector_sram"}

    # An authored allocation names its memory; the kind is not user input.
    explicit_rram = compound.replace(
        "on @SRAM {loom.inferred_residency}", "on @RRAM"
    )
    error, _, rram_etg = run(_loom_pipeline, explicit_rram, hardware)
    assert not error, error
    assert len(json.loads(rram_etg)) == 1
    assert {
        item["target"]["array"]
        for item in walk_placed(json.loads(rram_etg))
        if item["func"]["name"] in {"sub", "exp"}
    } == {"vector_rram"}

    conflicting_source = source.replace(
        "linalg.matmul ins(%a2, %b2 : memref<32x64xf16>, memref<64x32xf16>) outs(%c2 : memref<32x32xf32>)",
        "linalg.matmul ins(%a, %b : memref<32x64xf16>, memref<64x32xf16>) outs(%c : memref<32x32xf32>)",
    )
    conflict_error, conflict_explored, conflict_etg = run(
        _loom_pipeline, conflicting_source, hardware
    )
    assert not conflict_error, conflict_error
    assert len(json.loads(conflict_etg)) == 4
    assert "shared allocation has conflicting" in conflict_explored

    # Processor-definition metadata is optional for legacy hardware specs in
    # both fixed and enumerating binding modes.
    legacy_text = hardware_text.replace(
        ', mlar.processor_definition = "matrix"', ""
    ).replace(', mlar.processor_definition = "vector"', "")
    with tempfile.TemporaryDirectory() as temp_dir:
        legacy_hardware = Path(temp_dir) / "legacy.mlir"
        legacy_hardware.write_text(legacy_text)
        for enumerate in (False, True):
            legacy_error, _, legacy_etg = run(
                _loom_pipeline, compound, legacy_hardware,
                enumerate_bindings=enumerate,
            )
            assert not legacy_error, legacy_error
            assert json.loads(legacy_etg)

    # Missing direct movers remove candidates; no transfer chain is invented.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        filtered_hardware = temp_dir / "filtered.mlir"
        filtered_hardware.write_text(
            rewrite_mover_destination(hardware_text, "proc_dram_rram")
        )
        filtered_error, _, filtered_etg = run(
            _loom_pipeline, source, filtered_hardware
        )
        assert not filtered_error, filtered_error
        compute = [
            item
            for item in walk_placed(json.loads(filtered_etg))
            if item["func"]["name"] in {"matmul", "add"}
        ]
        assert {item["target"]["array"] for item in compute} == {
            "matrix_sram",
            "vector_sram",
        }

        fixed_error, _, fixed_etg = run(
            _loom_pipeline, source, filtered_hardware,
            enumerate_bindings=False,
        )
        assert not fixed_etg
        assert "fixed data-mover selection failed" in fixed_error

        unrouted_hardware = temp_dir / "unrouted.mlir"
        unrouted_hardware.write_text(
            rewrite_mover_destination(
                rewrite_mover_destination(hardware_text, "proc_dram_rram"),
                "proc_dram_sram",
            )
        )
        rejected, _, _ = run(_loom_pipeline, source, unrouted_hardware)
        assert "direct data-mover selection" in rejected

    # Real stage-00 Helion IR reaches all four bindings and materialization.
    dataflow_root = fixtures.parents[1]
    stage0 = (dataflow_root / "examples/mm/IR/00_from_helion_frontend.mlir").read_text()
    mesh = dataflow_root.parent / "loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir"
    error, explored, etg = run(
        _loom_pipeline,
        stage0,
        mesh,
        explicit_memory=False,
    )
    assert not error, error
    stage0_variants = json.loads(etg)
    compute_arrays = {
        item["target"]["array"]
        for variant in stage0_variants
        for item in walk_placed(variant)
        if item["func"]["name"] == "matmul_f16"
    }
    assert compute_arrays == {
        "matrix_lane_rr",
        "matrix_lane_rs",
        "matrix_lane_sr",
        "matrix_lane_ss",
    }
    selected = stage0_variants[0]["variant_name"]
    bindings = {
        "__loom_candidate_order__": [selected],
        selected: {
            "tile_m": 256,
            "tile_n": 32,
            "tile_k": 256,
            "is_double_buffer": 0,
        },
    }
    old_target = os.environ.get("LOOM_TARGET")
    os.environ["LOOM_TARGET"] = "tt"
    conflicting_target, _ = _loom_pipeline.run_materialization_pipeline(
        explored, json.dumps(bindings), str(mesh)
    )
    if old_target is None:
        del os.environ["LOOM_TARGET"]
    else:
        os.environ["LOOM_TARGET"] = old_target
    assert "artifact target 'generic' conflicts" in conflicting_target
    error, materialized = _loom_pipeline.run_materialization_pipeline(
        explored, json.dumps(bindings), str(mesh)
    )
    assert not error, error
    assert "loom.binding_site = 0" in materialized
    materialized_copies = [
        line for line in materialized.splitlines() if "loom.copy " in line
    ]
    assert len(materialized_copies) == 3
    assert all("loom.processor_array" in line for line in materialized_copies)

    old_target = os.environ.get("LOOM_TARGET")
    os.environ["LOOM_TARGET"] = "tt"
    try:
        tt_error, tt_explored, tt_etg = run(
            _loom_pipeline, stage0, mesh,
            explicit_memory=False, enumerate_bindings=False,
        )
        assert not tt_error, tt_error
        tt_variant = json.loads(tt_etg)[0]
        assert tt_variant["target"] == "tt"
        tt_values = {
            name: max(1, info.get("natural_ub", -1))
            for name, info in tt_variant["constraint_scope"]["metadata"]["symbols"].items()
        }
        tt_values["is_double_buffer"] = 0
        tt_error, tt_materialized = _loom_pipeline.run_materialization_pipeline(
            tt_explored,
            json.dumps({
                "__loom_candidate_order__": [tt_variant["variant_name"]],
                tt_variant["variant_name"]: tt_values,
            }),
            str(mesh),
        )
        assert not tt_error, tt_error
        assert 'loom.target = "tt"' in tt_materialized
        assert "loom.processor_function" in tt_materialized
    finally:
        if old_target is None:
            del os.environ["LOOM_TARGET"]
        else:
            os.environ["LOOM_TARGET"] = old_target

    # Both binding modes carry real matmul and FlashAttention templates through
    # ETG construction and selected-variant materialization.
    flash_source = (
        dataflow_root / "examples/flashattn/IR/00_from_helion_frontend.mlir"
    ).read_text()
    fixed_flash_error, _, _ = run(
        _loom_pipeline, flash_source, mesh,
        explicit_memory=False, enumerate_bindings=False,
    )
    assert "shared allocation has conflicting" in fixed_flash_error

    # Residency is requested by platform memory name and validated up front.
    pinned = 'memory = "L1_S"'
    for replacement, message in (
        ('memory = "L1_X"', "unknown residency memory 'L1_X'; platform memories: DRAM, L1_R, L1_S"),
        ('memory = "mem_L1_S"', "must be the platform memory name 'L1_S'"),
        ("local_mem_kind = 1 : i64", "numeric local_mem_kind is internal"),
        ('memory = "DRAM"', "satisfies the explicit residency (operand 1 -> DRAM)"),
    ):
        error, _, _ = run(
            _loom_pipeline, flash_source.replace(pinned, replacement), mesh,
            explicit_memory=False,
        )
        assert message in error, error

    def matmul_rhs_memories(source):
        error, explored, _ = run(_loom_pipeline, source, mesh, explicit_memory=False)
        assert not error, error
        text = explored.split('loom.binding_manifest = "', 1)[1].split('"', 1)[0]
        manifest = json.loads(
            re.sub(r"\\([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), text)
        )
        per_site = {}
        for entry in manifest:
            for selection in entry.get("selections", []):
                if selection["function"].startswith("batch_matmul"):
                    per_site.setdefault(selection["site"], set()).add(
                        selection["operand_memories"][1]
                    )
        return per_site

    # The global endpoint comes from the platform's DRAM-domain memory.
    mesh_text = mesh.read_text()
    with tempfile.TemporaryDirectory() as temp:
        for name, text, message in (
            ("no_domain.mlir", mesh_text.replace(' {domain = "DRAM"}', ""),
             "exactly one DRAM-domain memory"),
            ("reserved.mlir", mesh_text.replace("@mem_L1_S", "@mem___unbound"),
             "reserved memory name 'mem___unbound'"),
        ):
            platform = Path(temp) / name
            platform.write_text(text)
            error, _, _ = run(
                _loom_pipeline, flash_source.replace(", {" + pinned + "}", ""),
                platform, explicit_memory=False,
            )
            assert message in error, error

    # Pinning K narrows exactly its batch_matmul site; the PV site stays free.
    free = matmul_rhs_memories(flash_source.replace(", {" + pinned + "}", ""))
    pinned_sites = matmul_rhs_memories(flash_source)
    assert free.keys() == pinned_sites.keys() and len(free) == 2, (free, pinned_sites)
    narrowed = [site for site in free if free[site] != pinned_sites[site]]
    assert len(narrowed) == 1, (free, pinned_sites)
    assert free[narrowed[0]] == {"mem_L1_R", "mem_L1_S"}
    assert pinned_sites[narrowed[0]] == {"mem_L1_S"}

    for template, enumerate in ((stage0, False), (flash_source, True)):
        error, explored, etg = run(
            _loom_pipeline, template, mesh,
            explicit_memory=False, enumerate_bindings=enumerate,
        )
        assert not error, error
        variant = json.loads(etg)[0]
        selected = variant["variant_name"]
        symbols = variant["constraint_scope"]["metadata"]["symbols"]
        values = {
            name: max(1, info.get("natural_ub", -1))
            for name, info in symbols.items()
        }
        values["is_double_buffer"] = 0
        error, materialized = _loom_pipeline.run_materialization_pipeline(
            explored,
            json.dumps({"__loom_candidate_order__": [selected], selected: values}),
            str(mesh),
        )
        assert not error, error
        assert "loom.binding_site" in materialized
        assert "loom.processor_function" in materialized
        if template is flash_source:
            assert "arith.cmpf" in explored and "arith.select" in explored
            assert "math.exp" in materialized
            assert "mem_L1_S" in materialized
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
