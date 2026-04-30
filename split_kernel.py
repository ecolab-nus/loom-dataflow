#!/usr/bin/env python3
"""
Script to split kernel.cpp into separate files based on comment markers.
Each function section (identified by a comment line like "// foo__compute")
will be written to its own file.
"""

import argparse
import json
import os
import re
from pathlib import Path


def insert_include_if_missing(lines, include_line):
    if any(line.strip() == include_line.strip() for line in lines):
        return lines

    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("#include"):
            insert_idx = i + 1
    lines.insert(insert_idx, include_line)
    return lines


def insert_line_after_includes_if_missing(lines, new_line):
    if any(line.strip() == new_line.strip() for line in lines):
        return lines

    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("#include"):
            insert_idx = i + 1
    lines.insert(insert_idx, new_line)
    return lines


def insert_block_after_includes_if_missing(lines, block_lines):
    stripped_block = [line.strip() for line in block_lines]
    joined_window = "\n".join(line.strip() for line in lines)
    if "\n".join(stripped_block) in joined_window:
        return lines

    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("#include"):
            insert_idx = i + 1

    for offset, line in enumerate(block_lines):
        lines.insert(insert_idx + offset, line)
    return lines


def insert_compute_trace_markers(lines):
    """
    Insert DPRINT markers before cb_wait_front calls.

    N is a running counter across both call kinds in source order.
    """
    marker_pattern = re.compile(r"^\s*(cb_wait_front\s*\()")

    dprint_pattern = re.compile(r'^\s*DPRINT\s*<<\s*"compute"\s*<<')

    instrumented = []
    counter = 0
    for line in lines:
        if marker_pattern.search(line):
            if not (instrumented and dprint_pattern.search(instrumented[-1])):
                indent = line[: len(line) - len(line.lstrip())]
                if counter > 0:
                    instrumented.append(
                        f'{indent}DPRINT << "compute {counter}" << ENDL();\n'
                    )
                counter += 1
        instrumented.append(line)
    return instrumented


def _try_parse_u32_for_header(line):
    match = re.match(
        r"^\s*for\s*\(\s*uint32_t\s+(\w+)\s*=\s*([^;]+);\s*\1\s*<\s*([^;]+);\s*\1\s*\+=\s*([^)]+)\)\s*\{\s*$",
        line,
    )
    if not match:
        return None
    return {
        "var": match.group(1).strip(),
        "start": match.group(2).strip(),
        "limit": match.group(3).strip(),
        "step": match.group(4).strip(),
    }


def _extract_uint32_constant_map(lines):
    values = {}
    for line in lines:
        match = re.match(
            r"^\s*uint32_t\s+(\w+)\s*=\s*(-?(?:0[xX][0-9a-fA-F]+|\d+))\s*;\s*$",
            line,
        )
        if not match:
            continue
        symbol = match.group(1)
        literal = match.group(2)
        try:
            values[symbol] = int(literal, 0)
        except ValueError:
            continue
    return values


def _resolve_int_token(token, constants):
    tok = token.strip()
    if tok in constants:
        return constants[tok]
    try:
        return int(tok, 0)
    except ValueError:
        return None


def _is_zero_token(token, constants):
    value = _resolve_int_token(token, constants)
    return value == 0


def _is_one_token(token, constants):
    value = _resolve_int_token(token, constants)
    return value == 1


def _match_copy_block(lines, start_idx, constants):
    if start_idx + 10 >= len(lines):
        return None

    m_reserve = re.match(
        r"^(\s*)cb_reserve_back\(([^,]+),\s*([^)]+)\);\s*$", lines[start_idx]
    )
    if not m_reserve:
        return None
    indent = m_reserve.group(1)
    reserved_out_cb = m_reserve.group(2).strip()
    reserved_tile_count = m_reserve.group(3).strip()

    init_idx = start_idx + 1
    m_init = re.match(r"^\s*copy_tile_init\(([^)]+)\);\s*$", lines[init_idx])
    if not m_init:
        return None
    in_cb = m_init.group(1).strip()

    loop = _try_parse_u32_for_header(lines[init_idx + 1])
    if not loop:
        return None
    if loop["limit"] != reserved_tile_count:
        return None
    if not _is_zero_token(loop["start"], constants) or not _is_one_token(
        loop["step"], constants
    ):
        return None

    if lines[init_idx + 2].strip() != "tile_regs_acquire();":
        return None

    m_copy = re.match(
        r"^\s*copy_tile\(([^,]+),\s*([^,]+),\s*([^)]+)\);\s*$", lines[init_idx + 3]
    )
    if not m_copy:
        return None
    copy_in_cb = m_copy.group(1).strip()
    copy_tile_idx = m_copy.group(2).strip()
    dst_idx = m_copy.group(3).strip()
    if copy_in_cb != in_cb or copy_tile_idx != loop["var"]:
        return None

    if lines[init_idx + 4].strip() != "tile_regs_commit();":
        return None
    if lines[init_idx + 5].strip() != "tile_regs_wait();":
        return None

    m_pack = re.match(
        r"^\s*pack_tile<true>\(([^,]+),\s*([^,]+),\s*([^)]+)\);\s*$",
        lines[init_idx + 6],
    )
    if not m_pack:
        return None
    pack_dst = m_pack.group(1).strip()
    out_cb = m_pack.group(2).strip()
    pack_tile_idx = m_pack.group(3).strip()
    if (
        pack_dst != dst_idx
        or pack_tile_idx != loop["var"]
        or out_cb != reserved_out_cb
    ):
        return None

    if lines[init_idx + 7].strip() != "tile_regs_release();":
        return None
    if lines[init_idx + 8].strip() != "}":
        return None

    m_push = re.match(
        r"^\s*cb_push_back\(([^,]+),\s*([^)]+)\);\s*$", lines[init_idx + 9]
    )
    if not m_push:
        return None
    push_out = m_push.group(1).strip()
    push_tiles = m_push.group(2).strip()
    if push_out != out_cb or push_tiles != loop["limit"]:
        return None

    replacement = (
        f"{indent}loom_copy_block({in_cb}, {out_cb}, {loop['limit']}, {dst_idx});\n"
    )
    return {"end_idx": init_idx + 10, "replacement": replacement}


def _match_fill_block(lines, start_idx, constants):
    if start_idx + 11 >= len(lines):
        return None

    m_reserve = re.match(
        r"^(\s*)cb_reserve_back\(([^,]+),\s*([^)]+)\);\s*$", lines[start_idx]
    )
    if not m_reserve:
        return None
    indent = m_reserve.group(1)
    out_cb = m_reserve.group(2).strip()
    tile_count = m_reserve.group(3).strip()

    m_init = re.match(
        r"^\s*init_sfpu\(([^,]+),\s*([^)]+)\);\s*$", lines[start_idx + 1]
    )
    if not m_init:
        return None
    if m_init.group(1).strip() != out_cb or m_init.group(2).strip() != out_cb:
        return None

    if lines[start_idx + 2].strip() != "fill_tile_init();":
        return None

    loop = _try_parse_u32_for_header(lines[start_idx + 3])
    if not loop:
        return None
    if (
        loop["limit"] != tile_count
        or not _is_zero_token(loop["start"], constants)
        or not _is_one_token(loop["step"], constants)
    ):
        return None

    if lines[start_idx + 4].strip() != "tile_regs_acquire();":
        return None

    m_fill = re.match(
        r"^\s*fill_tile\(([^,]+),\s*([^)]+)\);\s*$", lines[start_idx + 5]
    )
    if not m_fill:
        return None
    dst_idx = m_fill.group(1).strip()
    fill_val = m_fill.group(2).strip()

    if lines[start_idx + 6].strip() != "tile_regs_commit();":
        return None
    if lines[start_idx + 7].strip() != "tile_regs_wait();":
        return None

    m_pack = re.match(
        r"^\s*pack_tile<true>\(([^,]+),\s*([^,]+),\s*([^)]+)\);\s*$", lines[start_idx + 8]
    )
    if not m_pack:
        return None
    pack_dst = m_pack.group(1).strip()
    pack_out = m_pack.group(2).strip()
    pack_tile_idx = m_pack.group(3).strip()
    if pack_dst != dst_idx or pack_out != out_cb or pack_tile_idx != loop["var"]:
        return None

    if lines[start_idx + 9].strip() != "tile_regs_release();":
        return None
    if lines[start_idx + 10].strip() != "}":
        return None

    m_push = re.match(
        r"^\s*cb_push_back\(([^,]+),\s*([^)]+)\);\s*$", lines[start_idx + 11]
    )
    if not m_push:
        return None
    if m_push.group(1).strip() != out_cb or m_push.group(2).strip() != tile_count:
        return None

    replacement = (
        f"{indent}loom_fill_block({out_cb}, {tile_count}, {fill_val}, {dst_idx});\n"
    )
    return {"end_idx": start_idx + 12, "replacement": replacement}


def _match_unary_bcast_block(lines, start_idx, constants):
    if start_idx + 11 >= len(lines):
        return None

    m_wait = re.match(
        r"^(\s*)cb_wait_front\(([^,]+),\s*([^)]+)\);\s*$", lines[start_idx]
    )
    if not m_wait:
        return None
    indent = m_wait.group(1)
    in_cb = m_wait.group(2).strip()
    tile_count = m_wait.group(3).strip()

    m_reserve = re.match(
        r"^\s*cb_reserve_back\(([^,]+),\s*([^)]+)\);\s*$", lines[start_idx + 1]
    )
    if not m_reserve:
        return None
    out_cb = m_reserve.group(1).strip()
    reserve_tiles = m_reserve.group(2).strip()
    if reserve_tiles != tile_count:
        return None

    m_init = re.match(
        r"^\s*unary_bcast_init<BroadcastType::(ROW|COL)>\(([^,]+),\s*([^)]+)\);\s*$",
        lines[start_idx + 2],
    )
    if not m_init:
        return None
    kind = m_init.group(1)
    init_in_cb = m_init.group(2).strip()
    init_out_cb = m_init.group(3).strip()
    if init_in_cb != in_cb or init_out_cb != out_cb:
        return None

    loop = _try_parse_u32_for_header(lines[start_idx + 3])
    if not loop:
        return None
    if loop["limit"] != tile_count:
        return None
    if not _is_zero_token(loop["start"], constants) or not _is_one_token(
        loop["step"], constants
    ):
        return None

    if lines[start_idx + 4].strip() != "tile_regs_acquire();":
        return None

    m_bcast = re.match(
        r"^\s*unary_bcast<BroadcastType::(ROW|COL)>\(([^,]+),\s*([^,]+),\s*([^)]+)\);\s*$",
        lines[start_idx + 5],
    )
    if not m_bcast:
        return None
    tile_kind = m_bcast.group(1)
    tile_in_cb = m_bcast.group(2).strip()
    tile_idx = m_bcast.group(3).strip()
    dst_idx = m_bcast.group(4).strip()
    if tile_kind != kind or tile_in_cb != in_cb or tile_idx != loop["var"]:
        return None

    if lines[start_idx + 6].strip() != "tile_regs_commit();":
        return None
    if lines[start_idx + 7].strip() != "tile_regs_wait();":
        return None

    m_pack = re.match(
        r"^\s*pack_tile<true>\(([^,]+),\s*([^,]+),\s*([^)]+)\);\s*$",
        lines[start_idx + 8],
    )
    if not m_pack:
        return None
    pack_dst = m_pack.group(1).strip()
    pack_out_cb = m_pack.group(2).strip()
    pack_tile_idx = m_pack.group(3).strip()
    if pack_dst != dst_idx or pack_out_cb != out_cb or pack_tile_idx != loop["var"]:
        return None

    if lines[start_idx + 9].strip() != "tile_regs_release();":
        return None
    if lines[start_idx + 10].strip() != "}":
        return None

    m_push = re.match(
        r"^\s*cb_push_back\(([^,]+),\s*([^)]+)\);\s*$", lines[start_idx + 11]
    )
    if not m_push:
        return None
    if m_push.group(1).strip() != out_cb or m_push.group(2).strip() != loop["limit"]:
        return None

    kind_token = (
        "LOOM_BCAST_KIND_ROW" if kind == "ROW" else "LOOM_BCAST_KIND_COL"
    )
    replacement = (
        f"{indent}loom_unary_bcast_block({in_cb}, {out_cb}, {loop['limit']}, {dst_idx}, {kind_token});\n"
    )
    return {"end_idx": start_idx + 12, "replacement": replacement}


def _match_inplace_cb_advance_block(lines, start_idx):
    if start_idx + 2 >= len(lines):
        return None

    m_pop = re.match(
        r"^(\s*)cb_pop_front\(([^,]+),\s*([^)]+)\);\s*$", lines[start_idx]
    )
    if not m_pop:
        return None
    indent = m_pop.group(1)
    cb_id = m_pop.group(2).strip()
    tile_count = m_pop.group(3).strip()

    m_reserve = re.match(
        r"^\s*cb_reserve_back\(([^,]+),\s*([^)]+)\);\s*$", lines[start_idx + 1]
    )
    if not m_reserve:
        return None
    reserve_cb = m_reserve.group(1).strip()
    reserve_tiles = m_reserve.group(2).strip()
    if reserve_cb != cb_id or reserve_tiles != tile_count:
        return None

    m_push = re.match(
        r"^\s*cb_push_back\(([^,]+),\s*([^)]+)\);\s*$", lines[start_idx + 2]
    )
    if not m_push:
        return None
    push_cb = m_push.group(1).strip()
    push_tiles = m_push.group(2).strip()
    if push_cb != cb_id or push_tiles != tile_count:
        return None

    replacement = f"{indent}loom_inplace_cb_advance({cb_id}, {tile_count});\n"
    return {"end_idx": start_idx + 3, "replacement": replacement}


def _match_commit_wait_pack_release_block(lines, start_idx):
    if start_idx + 3 >= len(lines):
        return None

    if lines[start_idx].strip() != "tile_regs_commit();":
        return None
    if lines[start_idx + 1].strip() != "tile_regs_wait();":
        return None

    m_pack = re.match(
        r"^(\s*)pack_tile<true>\(([^,]+),\s*([^,]+),\s*([^)]+)\);\s*$",
        lines[start_idx + 2],
    )
    if not m_pack:
        return None
    indent = m_pack.group(1)
    dst_idx = m_pack.group(2).strip()
    out_cb = m_pack.group(3).strip()
    tile_idx = m_pack.group(4).strip()

    if lines[start_idx + 3].strip() != "tile_regs_release();":
        return None

    replacement = (
        f"{indent}loom_commit_wait_pack_release({dst_idx}, {out_cb}, {tile_idx});\n"
    )
    return {"end_idx": start_idx + 4, "replacement": replacement}


def _compact_compute_blocks(lines):
    constants = _extract_uint32_constant_map(lines)
    compacted = []
    usage = {
        "copy": False,
        "fill": False,
        "bcast": False,
        "inplace": False,
        "pack_release": False,
    }

    i = 0
    while i < len(lines):
        fill = _match_fill_block(lines, i, constants)
        if fill:
            compacted.append(fill["replacement"])
            usage["fill"] = True
            i = fill["end_idx"]
            continue

        copy = _match_copy_block(lines, i, constants)
        if copy:
            compacted.append(copy["replacement"])
            usage["copy"] = True
            i = copy["end_idx"]
            continue

        bcast = _match_unary_bcast_block(lines, i, constants)
        if bcast:
            compacted.append(bcast["replacement"])
            usage["bcast"] = True
            i = bcast["end_idx"]
            continue

        inplace = _match_inplace_cb_advance_block(lines, i)
        if inplace:
            compacted.append(inplace["replacement"])
            usage["inplace"] = True
            i = inplace["end_idx"]
            continue

        pack_release = _match_commit_wait_pack_release_block(lines, i)
        if pack_release:
            compacted.append(pack_release["replacement"])
            usage["pack_release"] = True
            i = pack_release["end_idx"]
            continue

        compacted.append(lines[i])
        i += 1

    return compacted, usage


def _build_compute_helper_block(usage):
    block = []
    block.append("\n")
    if usage.get("bcast"):
        block.extend(
            [
                "constexpr uint32_t LOOM_BCAST_KIND_ROW = 0;\n",
                "constexpr uint32_t LOOM_BCAST_KIND_COL = 1;\n",
                "\n",
                "template <BroadcastType BcastKind>\n",
                "__attribute__((noinline)) void loom_unary_bcast_block_impl(\n",
                "    uint32_t in_cb, uint32_t out_cb, uint32_t tile_count,\n",
                "    uint32_t dst_idx) {\n",
                "  cb_wait_front(in_cb, tile_count);\n",
                "  cb_reserve_back(out_cb, tile_count);\n",
                "  unary_bcast_init<BcastKind>(in_cb, out_cb);\n",
                "  for (uint32_t i = 0; i < tile_count; i += 1) {\n",
                "    tile_regs_acquire();\n",
                "    unary_bcast<BcastKind>(in_cb, i, dst_idx);\n",
                "    tile_regs_commit();\n",
                "    tile_regs_wait();\n",
                "    pack_tile<true>(dst_idx, out_cb, i);\n",
                "    tile_regs_release();\n",
                "  }\n",
                "  cb_push_back(out_cb, tile_count);\n",
                "}\n",
                "\n",
                "__attribute__((noinline)) void loom_unary_bcast_block(\n",
                "    uint32_t in_cb, uint32_t out_cb, uint32_t tile_count,\n",
                "    uint32_t dst_idx, uint32_t bcast_kind) {\n",
                "  if (bcast_kind == LOOM_BCAST_KIND_ROW) {\n",
                "    loom_unary_bcast_block_impl<BroadcastType::ROW>(in_cb, out_cb,\n",
                "                                                     tile_count, dst_idx);\n",
                "    return;\n",
                "  }\n",
                "  loom_unary_bcast_block_impl<BroadcastType::COL>(in_cb, out_cb,\n",
                "                                                   tile_count, dst_idx);\n",
                "}\n",
                "\n",
            ]
        )

    if usage.get("copy"):
        block.extend(
            [
                "__attribute__((noinline)) void loom_copy_block(\n",
                "    uint32_t in_cb, uint32_t out_cb, uint32_t tile_count,\n",
                "    uint32_t dst_idx) {\n",
                "  cb_reserve_back(out_cb, tile_count);\n",
                "  copy_tile_init(in_cb);\n",
                "  for (uint32_t i = 0; i < tile_count; i += 1) {\n",
                "    tile_regs_acquire();\n",
                "    copy_tile(in_cb, i, dst_idx);\n",
                "    tile_regs_commit();\n",
                "    tile_regs_wait();\n",
                "    pack_tile<true>(dst_idx, out_cb, i);\n",
                "    tile_regs_release();\n",
                "  }\n",
                "  cb_push_back(out_cb, tile_count);\n",
                "}\n",
                "\n",
            ]
        )

    if usage.get("fill"):
        block.extend(
            [
                "__attribute__((noinline)) void loom_fill_block(\n",
                "    uint32_t out_cb, uint32_t tile_count, float fill_val,\n",
                "    uint32_t dst_idx) {\n",
                "  cb_reserve_back(out_cb, tile_count);\n",
                "  init_sfpu(out_cb, out_cb);\n",
                "  fill_tile_init();\n",
                "  for (uint32_t i = 0; i < tile_count; i += 1) {\n",
                "    tile_regs_acquire();\n",
                "    fill_tile(dst_idx, fill_val);\n",
                "    tile_regs_commit();\n",
                "    tile_regs_wait();\n",
                "    pack_tile<true>(dst_idx, out_cb, i);\n",
                "    tile_regs_release();\n",
                "  }\n",
                "  cb_push_back(out_cb, tile_count);\n",
                "}\n",
                "\n",
            ]
        )

    if usage.get("inplace"):
        block.extend(
            [
                "__attribute__((noinline)) void loom_inplace_cb_advance(\n",
                "    uint32_t cb_id, uint32_t tile_count) {\n",
                "  cb_pop_front(cb_id, tile_count);\n",
                "  cb_reserve_back(cb_id, tile_count);\n",
                "  cb_push_back(cb_id, tile_count);\n",
                "}\n",
                "\n",
            ]
        )

    if usage.get("pack_release"):
        block.extend(
            [
                "__attribute__((noinline)) void loom_commit_wait_pack_release(\n",
                "    uint32_t dst_idx, uint32_t out_cb, uint32_t tile_idx) {\n",
                "  tile_regs_commit();\n",
                "  tile_regs_wait();\n",
                "  pack_tile<true>(dst_idx, out_cb, tile_idx);\n",
                "  tile_regs_release();\n",
                "}\n",
                "\n",
            ]
        )

    return block


def process_source_content(lines, section_name=None):
    """
    Process source file content by applying necessary replacements.
    
    Args:
        lines: List of source code lines
    
    Returns:
        List of processed lines with replacements applied
    """
    processed = [
        line.replace("::tt::CB", "uint32_t")
        .replace('uint32_t', 'int32_t')
        .replace('int32_t', 'uint32_t')
        .replace('pack_tile<false>', 'pack_tile<true>')
        .replace("INFINITY", '100.0')
        .replace("tt_metal/programming_examples/", '')
        for line in lines
    ]

    if section_name and section_name.startswith("host_pybind"):
        processed = [
            line
            for line in processed
            if line.strip()
            not in {
                '#include "tools/profiler/kernel_profiler.hpp"',
                '#include "firmware_common.h"',
                '#include "dataflow_api.h"',
            }
        ]
        processed = [line.replace(" tt_metal::", " tt::tt_metal::").replace('CBIndex::', 'tt::CBIndex::') for line in lines]

    if section_name and section_name.startswith("reader"):
        processed = insert_include_if_missing(processed, '#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"\n')

    if section_name and section_name.startswith("host"):
        processed = insert_include_if_missing(
            processed, "#include <tt-metalium/host_api.hpp>\n"
        )
        processed = insert_include_if_missing(
            processed, "#include <tt-metalium/tensor_accessor_args.hpp>\n"
        )
        processed = insert_include_if_missing(
            processed, "#include <tt-metalium/buffer.hpp>\n"
        )
        processed = insert_block_after_includes_if_missing(
            processed,
            [
                "#ifndef OVERRIDE_KERNEL_PREFIX\n",
                '#define OVERRIDE_KERNEL_PREFIX ""\n',
                "#endif\n",
            ],
        )

    if section_name and section_name.startswith("host_pybind"):
        processed = insert_include_if_missing(
            processed, "#include <tt-metalium/constants.hpp>\n"
        )
        processed = insert_include_if_missing(
            processed, '#include "ttnn/operation.hpp"\n'
        )
        processed = insert_include_if_missing(processed, "#include <optional>\n")
        processed = insert_include_if_missing(processed, "#include <utility>\n")
        processed = insert_include_if_missing(processed, "#include <vector>\n")
        processed = insert_line_after_includes_if_missing(
            processed, "using namespace tt::constants;\n"
        )
        processed = insert_line_after_includes_if_missing(
            processed, "using namespace tt::tt_metal;\n"
        )
        processed = insert_line_after_includes_if_missing(
            processed, "namespace tt_metal = tt::tt_metal;\n"
        )

    if section_name and (
        section_name.startswith("host_cpp") or section_name == "host.cpp"
    ):
        processed = insert_include_if_missing(processed, "#include <vector>\n")

    if section_name and section_name.startswith("compute"):
        #it seems that math.h is not needed for compute kernels on blackhole machine
        #processed = insert_include_if_missing(processed, '#include "math.h"\n')
        processed = insert_include_if_missing(processed, '#include "debug/dprint.h"\n')
        processed = insert_include_if_missing(processed, '#include "debug/dprint_pages.h"\n')
        processed = insert_include_if_missing(processed, '#include "debug/dprint_tensix.h"\n')
        processed = [i.replace("mm_init", "ckernel::mm_init").replace("mm_block_init_short", "ckernel::mm_block_init_short") for i in processed]
        processed = [i.replace("exp_tile(", "exp_tile<true, true>(") for i in processed]

        processed, compaction_usage = _compact_compute_blocks(processed)
        if any(compaction_usage.values()):
            helper_block = _build_compute_helper_block(compaction_usage)
            processed = insert_block_after_includes_if_missing(processed, helper_block)

        #Don't need it now
        #processed = insert_compute_trace_markers(processed)

    return processed


def extract_marker_symbol(comment_line):
    name = comment_line.strip()
    if name.startswith("//"):
        name = name[2:].strip()
    return name


def make_program_factory_name(comment_line):
    name = extract_marker_symbol(comment_line)
    if name.endswith("__host_pybind"):
        name = name[: -len("__host_pybind")]
    name = re.sub(r"[^\w]", "_", name)
    return f"{name}_program_factory"


def transform_host_pybind_source(lines):
    if not lines:
        return lines

    signature_pattern = re.compile(r"^(\s*)void\s+kernel_main\s*\(")
    return_pattern = re.compile(r"^(\s*)return;\s*$")
    factory_name = make_program_factory_name(lines[0])

    transformed = []
    signature_rewritten = False
    return_indices = []

    for line in lines:
        if not signature_rewritten and signature_pattern.search(line):
            line = signature_pattern.sub(
                rf"\1tt::tt_metal::operation::ProgramWithCallbacks {factory_name}(",
                line,
                count=1,
            )
            signature_rewritten = True
        transformed.append(line)
        if return_pattern.match(line):
            return_indices.append(len(transformed) - 1)

    if return_indices:
        last_return_idx = return_indices[-1]
        indent = return_pattern.match(transformed[last_return_idx]).group(1)
        transformed[last_return_idx] = (
            f"{indent}return {{.program = std::move(program), "
            f".override_runtime_arguments_callback = "
            f"override_runtime_arguments_callback}};\n"
        )

    return transformed


def make_host_ttnn_filename(section_cpp_filename):
    if not section_cpp_filename.startswith("host_pybind") or not section_cpp_filename.endswith(".cpp"):
        return None
    suffix = section_cpp_filename[len("host_pybind") : -len(".cpp")]
    return f"host_ttnn{suffix}.py"


def _safe_eval_u32_expr(expr, symbols):
    rewritten = expr
    for name, value in symbols.items():
        pattern = r"\b" + re.escape(name) + r"\b"
        rewritten = re.sub(pattern, str(value), rewritten)
    rewritten = rewritten.replace("/", "//")
    if not re.fullmatch(r"[0-9\(\)\+\-\*\/\s]+", rewritten):
        raise ValueError(f"unsupported expression: {expr}")
    return int(eval(rewritten, {"__builtins__": {}}, {}))


def _parse_noc_index(noc_token):
    token = noc_token.split("::")[-1]
    if token in {"RISCV_0_default", "NOC_0"}:
        return 0
    if token in {"RISCV_1_default", "NOC_1"}:
        return 1
    return None


def _parse_data_movement_processor(processor_token):
    token = processor_token.split("::")[-1]
    if token == "RISCV_0":
        return 0
    if token == "RISCV_1":
        return 1
    return None


def _normalize_token(token):
    tok = token.strip()
    if tok.endswith(","):
        tok = tok[:-1].strip()
    return tok


def _strip_outer_cpp_casts(expr):
    rewritten = expr.strip()
    while True:
        prev = rewritten
        static_cast_match = re.match(r"^static_cast<\s*[^>]+\s*>\((.*)\)$", rewritten)
        if static_cast_match:
            rewritten = static_cast_match.group(1).strip()
            continue

        c_style_cast_match = re.match(
            r"^\(\s*(?:std::)?(?:u?int\d+_t|size_t)\s*\)\s*(.*)$", rewritten
        )
        if c_style_cast_match:
            rewritten = c_style_cast_match.group(1).strip()
            continue

        if rewritten == prev:
            break
    return rewritten


def _translate_cpp_expr_to_python(expr):
    rewritten = expr.strip().rstrip(";").strip()
    rewritten = _strip_outer_cpp_casts(rewritten)
    rewritten = re.sub(r"\b(\d+)[uUlL]+\b", r"\1", rewritten)
    rewritten = rewritten.replace("core.x", "core_x").replace("core.y", "core_y")
    rewritten = re.sub(r"\b([A-Za-z_]\w*)\.(x|y)\b", r"\1_\2", rewritten)
    rewritten = re.sub(r"\btrue\b", "True", rewritten)
    rewritten = re.sub(r"\bfalse\b", "False", rewritten)
    rewritten = re.sub(r"\bstd::", "", rewritten)
    rewritten = rewritten.replace("/", "//")
    return rewritten


def _extract_runtime_loop_indices(lines):
    runtime_blocks = {}
    runtime_decl_re = re.compile(
        r"std::vector<\s*uint32_t\s*>\s+"
        r"((?:reader|writer|compute)_runtime_args_for_core|runtime_args_for_core)"
        r"\s*=\s*\{"
    )

    idx = 0
    while idx < len(lines):
        match = runtime_decl_re.search(lines[idx])
        if not match:
            idx += 1
            continue

        var_name = match.group(1)
        end_idx = idx + 1
        while end_idx < len(lines) and lines[end_idx].strip() != "};":
            end_idx += 1
        if end_idx >= len(lines):
            raise ValueError(f"runtime args block for {var_name} is unterminated")

        runtime_blocks[var_name] = (idx, end_idx)
        idx = end_idx + 1

    if not runtime_blocks:
        raise ValueError("runtime args block not found")

    first_runtime_decl_idx = min(start for start, _ in runtime_blocks.values())
    core_loop_start_idx = None
    core_loop_re = re.compile(r"for\s*\(\s*const auto&\s+core\s*:\s*cores\s*\)\s*\{")
    for idx in range(first_runtime_decl_idx - 1, -1, -1):
        if core_loop_re.search(lines[idx]):
            core_loop_start_idx = idx
            break

    if core_loop_start_idx is None:
        raise ValueError("failed to locate for (const auto& core : cores) loop")

    return core_loop_start_idx, first_runtime_decl_idx, runtime_blocks


def _translate_runtime_prelude_to_python(prelude_lines, buffer_to_param=None):
    translated = []
    core_coord_vars = {}
    buffer_to_param = buffer_to_param or {}

    for raw_line in prelude_lines:
        line = raw_line.strip()
        if not line or line in {"{", "}"}:
            continue

        core_coord_match = re.match(
            r"CoreCoord\s+(\w+)\s*=\s*\{(.+?),\s*(.+)\}\s*;", line
        )
        if core_coord_match:
            name = core_coord_match.group(1)
            x_expr = _translate_cpp_expr_to_python(core_coord_match.group(2))
            y_expr = _translate_cpp_expr_to_python(core_coord_match.group(3))
            core_coord_vars[name] = (x_expr, y_expr)
            translated.append(f"{name}_x = int({x_expr})")
            translated.append(f"{name}_y = int({y_expr})")
            continue

        physical_core_match = re.match(
            r"auto\s+(\w+)\s*=\s*device->worker_core_from_logical_core\((\w+)\)\s*;",
            line,
        )
        if physical_core_match:
            physical_name = physical_core_match.group(1)
            logical_name = physical_core_match.group(2)
            if logical_name not in core_coord_vars:
                raise ValueError(
                    f"unsupported runtime prelude: unknown CoreCoord {logical_name}"
                )
            translated.append(
                f"{physical_name}_x, {physical_name}_y = _logical_to_worker_core(device, int({logical_name}_x), int({logical_name}_y))"
            )
            continue

        vector_decl_match = re.match(
            r"std::vector<\s*bfloat16\s*>\s+(\w+)\s*;\s*$", line
        )
        if vector_decl_match:
            continue

        scalar_read_match = re.match(
            r"EnqueueReadSubBuffer\(cq,\s*\*(\w+),\s*(\w+),\s*"
            r"BufferRegion\(static_cast<DeviceAddr>\((.+?)\),\s*"
            r"static_cast<DeviceAddr>\(sizeof\(bfloat16\)\)\),\s*true\);\s*$",
            line,
        )
        if scalar_read_match:
            buffer_name = scalar_read_match.group(1)
            values_name = scalar_read_match.group(2)
            byte_expr = scalar_read_match.group(3).strip()
            element_expr = re.sub(
                r"\s*\*\s*sizeof\(bfloat16\)\s*$", "", byte_expr
            )
            element_expr = _translate_cpp_expr_to_python(element_expr)
            if buffer_name not in buffer_to_param:
                raise ValueError(
                    f"unsupported runtime prelude: unknown scalar buffer {buffer_name}"
                )
            param_name = buffer_to_param[buffer_name]
            translated.append(
                f"{values_name} = [_read_bfloat16_scalar(param_bindings[{json.dumps(param_name)}], int({element_expr}))]"
            )
            continue

        scalar_value_match = re.match(
            r"bfloat16\s+(\w+)\s*=\s*(\w+)\.front\(\)\s*;\s*$", line
        )
        if scalar_value_match:
            translated.append(
                f"{scalar_value_match.group(1)} = {scalar_value_match.group(2)}[0]"
            )
            continue

        scalar_table_value_match = re.match(
            r"bfloat16\s+(\w+)\s*=\s*(\w+)\.at\((.+)\)\s*;\s*$", line
        )
        if scalar_table_value_match:
            value_name = scalar_table_value_match.group(1)
            table_name = scalar_table_value_match.group(2)
            index_expr = _translate_cpp_expr_to_python(
                scalar_table_value_match.group(3)
            )
            translated.append(f"{value_name} = {table_name}[int({index_expr})]")
            continue

        scalar_pack_match = re.match(
            r"uint32_t\s+(\w+)\s*=\s*pack_two_bfloat16_into_uint32\(\{\s*(\w+)\s*,\s*(\w+)\s*\}\)\s*;\s*$",
            line,
        )
        if scalar_pack_match:
            translated.append(
                f"{scalar_pack_match.group(1)} = _pack_two_bfloat16_into_uint32({scalar_pack_match.group(2)}, {scalar_pack_match.group(3)})"
            )
            continue

        int_decl_match = re.match(
            r"((?:std::)?(?:size_t|u?int(?:32|64)_t)|bool)\s+(\w+)\s*=\s*(.+)\s*;\s*$",
            line,
        )
        if int_decl_match:
            decl_type = int_decl_match.group(1)
            name = int_decl_match.group(2)
            expr = _translate_cpp_expr_to_python(int_decl_match.group(3))
            if decl_type.endswith("bool"):
                translated.append(f"{name} = bool({expr})")
            else:
                translated.append(f"{name} = int({expr})")
            continue

        raise ValueError(f"unsupported runtime prelude line: {line}")

    return translated


def _translate_runtime_token_to_python_expr(token, buffer_to_param, semaphore_id_map):
    tok = _strip_outer_cpp_casts(_normalize_token(token))

    cb_match = re.fullmatch(r"(?:tt::)?CBIndex::c_(\d+)", tok)
    if cb_match:
        return str(int(cb_match.group(1)))

    addr_match = re.fullmatch(r"(\w+)->address\(\)", tok)
    if addr_match:
        buffer_name = addr_match.group(1)
        if buffer_name not in buffer_to_param:
            raise ValueError(f"unknown buffer in runtime args: {buffer_name}")
        param_name = buffer_to_param[buffer_name]
        return f"_tensor_buffer_address(param_bindings[{json.dumps(param_name)}])"

    int_match = re.fullmatch(r"(\d+)(?:[uUlL]+)?", tok)
    if int_match:
        return str(int(int_match.group(1)))

    if tok in semaphore_id_map:
        return f"int(_SEMAPHORE_IDS[{json.dumps(tok)}])"

    if tok == "core.x":
        return "int(core_x)"
    if tok == "core.y":
        return "int(core_y)"

    return _translate_cpp_expr_to_python(tok)


def _product_of_int_tokens(text):
    factors = [int(token) for token in re.findall(r"\d+", text)]
    if not factors:
        return None
    product = 1
    for factor in factors:
        product *= factor
    return product


def _parse_wrapper_mesh_extents(wrapper_base):
    mesh_match = re.search(r"__x([0-9x]+)_y([0-9y]+)__", wrapper_base)
    if not mesh_match:
        return None, None
    return (
        _product_of_int_tokens(mesh_match.group(1)),
        _product_of_int_tokens(mesh_match.group(2)),
    )


def _extract_host_ttnn_metadata(lines):
    if not lines:
        raise ValueError("empty host_pybind section")

    wrapper_base = extract_marker_symbol(lines[0])
    if wrapper_base.endswith("__host_pybind"):
        wrapper_base = wrapper_base[: -len("__host_pybind")]
    mlir_core_extent_x, mlir_core_extent_y = _parse_wrapper_mesh_extents(wrapper_base)
    for line in lines:
        end_core_match = re.search(
            r"uint32_t\s+end_core_([xy])\s*=\s*(\d+)(?:[uUlL]+)?\s*;", line
        )
        if not end_core_match:
            continue
        extent = int(end_core_match.group(2)) + 1
        if end_core_match.group(1) == "x":
            mlir_core_extent_x = extent
        else:
            mlir_core_extent_y = extent
    wrapper_name = re.sub(r"[^\w]", "_", wrapper_base)
    if not wrapper_name:
        wrapper_name = "generated_kernel"
    if wrapper_name[0].isdigit():
        wrapper_name = f"k_{wrapper_name}"

    buffer_bindings = []
    for line in lines:
        match = re.search(r"auto\s*\*\s*(\w+)\s*=\s*v(\d+)\.buffer\(\);", line)
        if not match:
            continue
        buffer_name = match.group(1)
        arg_index = int(match.group(2)) - 1
        buffer_bindings.append((arg_index, buffer_name))

    if not buffer_bindings:
        raise ValueError("no tensor buffer bindings found")
    buffer_bindings.sort(key=lambda item: item[0])

    scalar_bindings = []
    for line in lines:
        match = re.search(
            r"const\s+auto\s*&\s*(\w+_scalar_values)\s*=\s*v(\d+)\s*;", line
        )
        if not match:
            continue
        scalar_bindings.append((int(match.group(2)) - 1, match.group(1)))
    scalar_bindings.sort(key=lambda item: item[0])

    host_bindings = []
    buffer_to_param = {}
    for _, buffer_name in buffer_bindings:
        param = buffer_name
        if param.endswith("_dram_buffer"):
            param = param[: -len("_dram_buffer")]
        buffer_to_param[buffer_name] = param
    for arg_index, buffer_name in buffer_bindings:
        host_bindings.append((arg_index, "buffer", buffer_name))
    for arg_index, scalar_name in scalar_bindings:
        host_bindings.append((arg_index, "scalar", scalar_name))
    host_bindings.sort(key=lambda item: item[0])

    param_order = []
    param_roles = {}
    for _, binding_kind, binding_name in host_bindings:
        if binding_kind == "scalar":
            param = binding_name
            if param in param_roles:
                continue
            param_order.append(param)
            param_roles[param] = "scalar"
            continue

        param = buffer_to_param[binding_name]
        param_order.append(param)
        if param.startswith("src"):
            param_roles[param] = "input"
        elif param.startswith("dst"):
            param_roles[param] = "output"
        elif param.startswith("io"):
            param_roles[param] = "io"
        else:
            param_roles[param] = "input"

    input_param_order = [
        param for param in param_order if param_roles.get(param) in {"input", "io"}
    ]
    output_param_order = [
        param for param in param_order if param_roles.get(param) in {"output", "io"}
    ]
    if not output_param_order and param_order:
        output_param_order = [param_order[-1]]

    semaphore_vars = []
    semaphore_initial_values = []
    semaphore_constants = {
        "INVALID": 0,
        "VALID": 1,
        "UINT32_MAX": (1 << 32) - 1,
    }
    for line in lines:
        match = re.search(
            r"auto\s+(\w+)\s*=\s*.*CreateSemaphore\([^,]+,\s*[^,]+,\s*([^)]+)\)",
            line,
        )
        if match:
            semaphore_vars.append(match.group(1))
            initial_token = match.group(2).strip()
            if initial_token in semaphore_constants:
                semaphore_initial_values.append(semaphore_constants[initial_token])
            else:
                semaphore_initial_values.append(int(initial_token, 0))
    semaphore_id_map = {name: idx for idx, name in enumerate(semaphore_vars)}

    cb_tile_exprs = {}
    for line in lines:
        match = re.search(r"const uint32_t\s+(\w+)\s*=\s*(.+);", line)
        if not match:
            continue
        const_name = match.group(1)
        const_expr = match.group(2).strip()
        if const_name.startswith("cb_tiles_per_block_"):
            cb_tile_exprs[const_name] = const_expr

    symbol_values = {
        "TILE_HEIGHT": 32,
        "TILE_WIDTH": 32,
        "single_tile_size": 2 * 1024,
        "cb_buffer_depth": 2,
    }
    for const_name, const_expr in cb_tile_exprs.items():
        symbol_values[const_name] = _safe_eval_u32_expr(const_expr, symbol_values)

    cb_layouts = []
    seen_cb_indices = set()
    for line in lines:
        match = re.search(
            r"CircularBufferConfig\((.+?),\s*\{\{(.+?)\}\}\)\.set_page_size",
            line,
        )
        if not match:
            continue
        total_expr = match.group(1).strip()
        cb_entries = match.group(2)
        total_size = _safe_eval_u32_expr(total_expr, symbol_values)
        cb_indices = []
        for cb_index_text in re.findall(r"(?:tt::)?CBIndex::c_(\d+)", cb_entries):
            cb_index = int(cb_index_text)
            if cb_index in seen_cb_indices:
                continue
            seen_cb_indices.add(cb_index)
            cb_indices.append(cb_index)
        if cb_indices:
            cb_layouts.append(
                {
                    "cb_indices": cb_indices,
                    "total_size": total_size,
                    "page_size": symbol_values["single_tile_size"],
                }
            )
    cb_layouts.sort(key=lambda item: item["cb_indices"][0])

    compile_arg_order = []
    for line in lines:
        match = re.search(
            r"TensorAccessorArgs\(\*?(\w+)\)\.append_to\(compile_args\);", line
        )
        if not match:
            continue
        buffer_name = match.group(1)
        if buffer_name not in buffer_to_param:
            continue
        compile_arg_order.append(buffer_to_param[buffer_name])

    kernel_sources = {}
    kernel_noc_indices = {}
    kernel_processors = {}
    for line in lines:
        match = re.search(
            r'auto\s+(\w+)\s*=\s*.*CreateKernel\(program,\s*OVERRIDE_KERNEL_PREFIX\s*"([^"]+)"',
            line,
        )
        if not match:
            continue
        kernel_id_var = match.group(1)
        kernel_sources[kernel_id_var] = match.group(2)
        noc_match = re.search(r"\.noc\s*=\s*((?:tt::tt_metal::)?NOC::\w+)", line)
        if noc_match:
            kernel_noc_indices[kernel_id_var] = _parse_noc_index(noc_match.group(1))
        processor_match = re.search(
            r"\.processor\s*=\s*((?:tt::tt_metal::)?DataMovementProcessor::\w+)",
            line,
        )
        if processor_match:
            kernel_processors[kernel_id_var] = _parse_data_movement_processor(processor_match.group(1))

    reader_kernel = kernel_sources.get("reader_id")
    writer_kernel = kernel_sources.get("writer_id")
    compute_kernel = kernel_sources.get("compute_kernel_id")
    if not reader_kernel or not writer_kernel or not compute_kernel:
        raise ValueError("missing one or more kernel source paths")

    core_loop_start, first_runtime_decl, runtime_blocks = (
        _extract_runtime_loop_indices(lines)
    )
    runtime_prelude_lines = lines[core_loop_start + 1 : first_runtime_decl]

    def collect_runtime_tokens(var_name):
        if var_name not in runtime_blocks:
            return None
        runtime_decl, runtime_end = runtime_blocks[var_name]
        tokens = []
        for raw_line in lines[runtime_decl + 1 : runtime_end]:
            token = _normalize_token(raw_line)
            if token:
                tokens.append(token)
        return tokens

    shared_runtime_tokens = collect_runtime_tokens("runtime_args_for_core")
    if shared_runtime_tokens is not None:
        runtime_tokens_by_role = {
            "reader": shared_runtime_tokens,
            "writer": shared_runtime_tokens,
            "compute": shared_runtime_tokens,
        }
    else:
        runtime_tokens_by_role = {
            "reader": collect_runtime_tokens("reader_runtime_args_for_core") or [],
            "writer": collect_runtime_tokens("writer_runtime_args_for_core") or [],
            "compute": collect_runtime_tokens("compute_runtime_args_for_core") or [],
        }

    runtime_prelude_python = _translate_runtime_prelude_to_python(
        runtime_prelude_lines, buffer_to_param
    )
    runtime_arg_exprs_by_role = {
        role: [
            _translate_runtime_token_to_python_expr(
                token, buffer_to_param, semaphore_id_map
            )
            for token in tokens
        ]
        for role, tokens in runtime_tokens_by_role.items()
    }

    return {
        "wrapper_name": wrapper_name,
        "mlir_core_extent_x": mlir_core_extent_x,
        "mlir_core_extent_y": mlir_core_extent_y,
        "param_order": param_order,
        "input_param_order": input_param_order,
        "output_param_order": output_param_order,
        "compile_arg_order": compile_arg_order,
        "cb_layouts": cb_layouts,
        "semaphore_id_map": semaphore_id_map,
        "semaphore_initial_values": semaphore_initial_values,
        "runtime_prelude_python": runtime_prelude_python,
        "runtime_arg_exprs_by_role": runtime_arg_exprs_by_role,
        "runtime_arg_exprs": runtime_arg_exprs_by_role["reader"],
        "reader_kernel": reader_kernel,
        "writer_kernel": writer_kernel,
        "compute_kernel": compute_kernel,
        "kernel_noc_indices": {
            "reader": kernel_noc_indices.get("reader_id"),
            "writer": kernel_noc_indices.get("writer_id"),
            "compute": kernel_noc_indices.get("compute_kernel_id"),
        },
        "kernel_dm_configs": {
            "reader": {
                "processor": kernel_processors.get("reader_id"),
                "noc": kernel_noc_indices.get("reader_id"),
            },
            "writer": {
                "processor": kernel_processors.get("writer_id"),
                "noc": kernel_noc_indices.get("writer_id"),
            },
        },
    }


def generate_host_ttnn_source(host_pybind_lines, source_cpp_filename):
    metadata = _extract_host_ttnn_metadata(host_pybind_lines)
    wrapper_fn = f"{metadata['wrapper_name']}_ttnn"
    param_sig = ", ".join(metadata["param_order"])

    source = []
    source.append("# Auto-generated by split_kernel.py from host_pybind.cpp.\n")
    source.append(f"# Source section: {source_cpp_filename}\n")
    source.append("from pathlib import Path\n")
    source.append("import re\n")
    source.append("import ttnn\n\n")
    source.append("UINT32_MAX = (1 << 32) - 1\n")
    source.append("SINGLE_TILE_SIZE = 2 * 1024\n\n")
    source.append(f"_PARAM_ORDER = {json.dumps(metadata['param_order'])}\n")
    source.append(f"_INPUT_PARAM_ORDER = {json.dumps(metadata['input_param_order'])}\n")
    source.append(f"_OUTPUT_PARAM_ORDER = {json.dumps(metadata['output_param_order'])}\n")
    source.append(f"_COMPILE_ARG_ORDER = {json.dumps(metadata['compile_arg_order'])}\n")
    source.append(f"_CB_LAYOUTS = {repr(metadata['cb_layouts'])}\n")
    source.append(f"_SEMAPHORE_IDS = {repr(metadata['semaphore_id_map'])}\n")
    source.append(f"_SEMAPHORE_INITIAL_VALUES = {repr(metadata['semaphore_initial_values'])}\n")
    source.append("SEMAPHORE_COUNT = len(_SEMAPHORE_IDS)\n")
    source.append(
        f"_KERNEL_SOURCES = {repr({'reader': metadata['reader_kernel'], 'writer': metadata['writer_kernel'], 'compute': metadata['compute_kernel']})}\n\n"
    )
    source.append(f"_KERNEL_SOURCE_NOC_INDEX = {repr(metadata['kernel_noc_indices'])}\n")
    source.append("_KERNEL_NOC_INDEX = {'reader': 0, 'writer': 1, 'compute': None}\n\n")
    source.append("def _source_type(name):\n")
    source.append("    if hasattr(ttnn.KernelDescriptor, \"SourceType\"):\n")
    source.append("        return getattr(ttnn.KernelDescriptor.SourceType, name)\n")
    source.append("    return getattr(ttnn._ttnn.program_descriptor.SourceType, name)\n\n")
    source.append("def _resolve_kernel_source_path(kernel_source):\n")
    source.append("    source_path = Path(kernel_source)\n")
    source.append("    candidates = [source_path, Path.cwd() / source_path]\n")
    source.append("    for parent in Path(__file__).resolve().parents:\n")
    source.append("        candidates.append(parent / source_path)\n")
    source.append("    for candidate in candidates:\n")
    source.append("        if candidate.exists():\n")
    source.append("            return candidate\n")
    source.append("    raise FileNotFoundError(kernel_source)\n\n")
    source.append("def _rewrite_explicit_noc_constants_for_descriptor(source, kernel_role, noc_index):\n")
    source.append("    source_noc_index = _KERNEL_SOURCE_NOC_INDEX.get(kernel_role)\n")
    source.append("    if source_noc_index is None or int(source_noc_index) == int(noc_index):\n")
    source.append("        return source\n")
    source.append("    old_value = int(source_noc_index)\n")
    source.append("    declaration_pattern = re.compile(rf\"\\b((?:u)?int8_t)\\s+([A-Za-z_]\\w*)\\s*=\\s*{old_value}\\s*;\")\n")
    source.append("    for declaration in list(declaration_pattern.finditer(source)):\n")
    source.append("        type_name, symbol = declaration.group(1), declaration.group(2)\n")
    source.append("        noc_use_pattern = re.compile(\n")
    source.append("            rf\"(?:get_noc_multicast_addr|noc_async_write_multicast|noc_semaphore_inc)\\([^;]*\\b{re.escape(symbol)}\\b\"\n")
    source.append("        )\n")
    source.append("        if not noc_use_pattern.search(source):\n")
    source.append("            continue\n")
    source.append("        return (\n")
    source.append("            source[: declaration.start()]\n")
    source.append("            + f\"{type_name} {symbol} = noc_index;\"\n")
    source.append("            + source[declaration.end() :]\n")
    source.append("        )\n")
    source.append("    return source\n\n")
    source.append("def _kernel_source_and_type(kernel_role):\n")
    source.append("    kernel_source = _KERNEL_SOURCES[kernel_role]\n")
    source.append("    noc_index = _KERNEL_NOC_INDEX.get(kernel_role)\n")
    source.append("    if noc_index is None:\n")
    source.append("        return kernel_source, _source_type(\"FILE_PATH\")\n")
    source.append("    source_path = _resolve_kernel_source_path(kernel_source)\n")
    source.append("    source = source_path.read_text()\n")
    source.append("    source = _rewrite_explicit_noc_constants_for_descriptor(source, kernel_role, noc_index)\n")
    source.append("    prefix = f\"#undef NOC_INDEX\\n#define NOC_INDEX {int(noc_index)}\\n\"\n")
    source.append("    return prefix + source, _source_type(\"SOURCE_CODE\")\n\n")
    source.append("def _u32(value):\n")
    source.append("    return int(value) & UINT32_MAX\n\n")
    source.append("def _tensor_buffer_address(tensor):\n")
    source.append("    if hasattr(tensor, \"buffer_address\"):\n")
    source.append("        return int(tensor.buffer_address())\n")
    source.append("    if hasattr(tensor, \"buffer\"):\n")
    source.append("        return int(tensor.buffer().address())\n")
    source.append("    raise TypeError(\"Tensor does not expose buffer address\")\n\n")
    source.append("def _logical_to_worker_core(device, core_x, core_y):\n")
    source.append("    logical_core = ttnn.CoreCoord(int(core_x), int(core_y))\n")
    source.append("    if hasattr(device, \"worker_core_from_logical_core\"):\n")
    source.append("        physical = device.worker_core_from_logical_core(logical_core)\n")
    source.append("        return int(physical.x), int(physical.y)\n")
    source.append("    return int(logical_core.x), int(logical_core.y)\n\n")
    source.append("def _read_bfloat16_scalar(tensor, index):\n")
    source.append("    import torch\n")
    source.append("    try:\n")
    source.append("        host_tensor = ttnn.from_device(tensor)\n")
    source.append("    except Exception:\n")
    source.append("        host_tensor = tensor\n")
    source.append("    if hasattr(ttnn, \"to_torch\"):\n")
    source.append("        torch_tensor = ttnn.to_torch(host_tensor)\n")
    source.append("    elif hasattr(host_tensor, \"to_torch\"):\n")
    source.append("        torch_tensor = host_tensor.to_torch()\n")
    source.append("    else:\n")
    source.append("        torch_tensor = torch.as_tensor(host_tensor)\n")
    source.append("    return torch_tensor.reshape(-1)[int(index)]\n\n")
    source.append("def _bfloat16_word(value):\n")
    source.append("    import torch\n")
    source.append("    tensor = torch.as_tensor([value], dtype=torch.bfloat16)\n")
    source.append("    return int(tensor.view(torch.int16)[0].item()) & 0xFFFF\n\n")
    source.append("def _pack_two_bfloat16_into_uint32(low, high):\n")
    source.append("    return _bfloat16_word(low) | (_bfloat16_word(high) << 16)\n\n")
    source.append(f"def {wrapper_fn}({param_sig}):\n")
    source.append("    param_bindings = {\n")
    for param in metadata["param_order"]:
        source.append(f"        \"{param}\": {param},\n")
    source.append("    }\n")
    source.append("    if not _PARAM_ORDER:\n")
    source.append("        raise ValueError(\"No memref-backed parameters were generated\")\n")
    first_param = metadata["param_order"][0]
    source.append(f"    device = param_bindings[\"{first_param}\"].device()\n")
    source.append("    core_grid = device.compute_with_storage_grid_size()\n")
    if metadata["mlir_core_extent_x"] is not None and metadata["mlir_core_extent_x"] > 0:
        source.append("    start_core_x = 0\n")
        source.append(f"    end_core_x = {metadata['mlir_core_extent_x'] - 1}\n")
    else:
        source.append("    start_core_x = 0\n")
        source.append("    end_core_x = int(core_grid.x - 1)\n")
    if metadata["mlir_core_extent_y"] is not None and metadata["mlir_core_extent_y"] > 0:
        source.append("    start_core_y = 0\n")
        source.append(f"    end_core_y = {metadata['mlir_core_extent_y'] - 1}\n")
    else:
        source.append("    start_core_y = 0\n")
        source.append("    end_core_y = int(core_grid.y - 1)\n")
    source.append(
        "    if end_core_x < start_core_x or end_core_y < start_core_y:\n"
    )
    source.append("        raise ValueError(\"invalid core range\")\n")
    source.append(
        "    if end_core_x >= int(core_grid.x) or end_core_y >= int(core_grid.y):\n"
    )
    source.append("        raise ValueError(\"core range exceeds device grid\")\n")
    source.append(
        "    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(start_core_x, start_core_y), ttnn.CoreCoord(end_core_x, end_core_y))])\n"
    )
    source.append("\n")
    source.append("    cbs = []\n")
    source.append("    for cb_layout in _CB_LAYOUTS:\n")
    source.append("        cb_formats = [\n")
    source.append("            ttnn.CBFormatDescriptor(\n")
    source.append("                buffer_index=int(cb_index),\n")
    source.append("                data_format=ttnn.bfloat16,\n")
    source.append("                page_size=int(cb_layout[\"page_size\"]),\n")
    source.append("            )\n")
    source.append("            for cb_index in cb_layout[\"cb_indices\"]\n")
    source.append("        ]\n")
    source.append("        cbs.append(\n")
    source.append("            ttnn.CBDescriptor(\n")
    source.append("                total_size=int(cb_layout[\"total_size\"]),\n")
    source.append("                core_ranges=core_ranges,\n")
    source.append("                format_descriptors=cb_formats,\n")
    source.append("            )\n")
    source.append("        )\n")
    source.append("\n")
    source.append(
        "    semaphores = [ttnn.SemaphoreDescriptor(core_ranges=core_ranges, initial_value=int(initial_value)) for initial_value in _SEMAPHORE_INITIAL_VALUES]\n"
    )
    source.append("\n")
    source.append("    compile_args = []\n")
    source.append("    for param_name in _COMPILE_ARG_ORDER:\n")
    source.append(
        "        compile_args.extend(ttnn.TensorAccessorArgs(param_bindings[param_name]).get_compile_time_args())\n"
    )
    source.append("\n")
    source.append("    reader_source, reader_source_type = _kernel_source_and_type(\"reader\")\n")
    source.append("    writer_source, writer_source_type = _kernel_source_and_type(\"writer\")\n")
    source.append("    compute_source, compute_source_type = _kernel_source_and_type(\"compute\")\n")
    source.append("\n")
    for role in ("reader", "writer", "compute"):
        source.append(f"    {role}_runtime_args = [\n")
        source.append("        [[] for _ in range(int(core_grid.y))]\n")
        source.append("        for _ in range(int(core_grid.x))\n")
        source.append("    ]\n")
    source.append("    for core_x in range(start_core_x, end_core_x + 1):\n")
    source.append("        for core_y in range(start_core_y, end_core_y + 1):\n")
    source.append("            core = ttnn.CoreCoord(int(core_x), int(core_y))\n")
    for line in metadata["runtime_prelude_python"]:
        source.append(f"            {line}\n")
    for role in ("reader", "writer", "compute"):
        source.append(f"            {role}_runtime_args_for_core = [\n")
        for expr in metadata["runtime_arg_exprs_by_role"].get(role, []):
            source.append(f"                _u32({expr}),\n")
        source.append("            ]\n")
        source.append(
            f"            {role}_runtime_args[core_x][core_y] = {role}_runtime_args_for_core\n"
        )
    source.append("\n")
    source.append("    reader_kernel_descriptor = ttnn.KernelDescriptor(\n")
    source.append("        kernel_source=reader_source,\n")
    source.append("        source_type=reader_source_type,\n")
    source.append("        core_ranges=core_ranges,\n")
    source.append("        compile_time_args=list(compile_args),\n")
    source.append("        runtime_args=reader_runtime_args,\n")
    source.append("        config=ttnn.ReaderConfigDescriptor(),\n")
    source.append("    )\n")
    source.append("    writer_kernel_descriptor = ttnn.KernelDescriptor(\n")
    source.append("        kernel_source=writer_source,\n")
    source.append("        source_type=writer_source_type,\n")
    source.append("        core_ranges=core_ranges,\n")
    source.append("        compile_time_args=list(compile_args),\n")
    source.append("        runtime_args=writer_runtime_args,\n")
    source.append("        config=ttnn.WriterConfigDescriptor(),\n")
    source.append("    )\n")
    source.append("    compute_kernel_descriptor = ttnn.KernelDescriptor(\n")
    source.append("        kernel_source=compute_source,\n")
    source.append("        source_type=compute_source_type,\n")
    source.append("        core_ranges=core_ranges,\n")
    source.append("        compile_time_args=list(compile_args),\n")
    source.append("        runtime_args=compute_runtime_args,\n")
    source.append("        config=ttnn.ComputeConfigDescriptor(),\n")
    source.append("    )\n")
    source.append("\n")
    source.append("    program_descriptor = ttnn.ProgramDescriptor(\n")
    source.append(
        "        kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],\n"
    )
    source.append("        semaphores=semaphores,\n")
    source.append("        cbs=cbs,\n")
    source.append("    )\n")
    source.append("\n")
    source.append("    input_tensors = [param_bindings[name] for name in _INPUT_PARAM_ORDER]\n")
    source.append("    output_tensors = [param_bindings[name] for name in _OUTPUT_PARAM_ORDER]\n")
    source.append("    io_tensors = input_tensors + output_tensors\n")
    source.append("    ttnn.generic_op(io_tensors, program_descriptor)\n")
    source.append("    if len(output_tensors) == 1:\n")
    source.append("        return output_tensors[0]\n")
    source.append("    return tuple(output_tensors)\n\n")
    source.append(f"run = {wrapper_fn}\n")
    return source


def _write_section_outputs(output_path, current_filename, current_section):
    output_file = output_path / current_filename
    processed_section = process_source_content(current_section, current_filename)
    if current_filename.startswith("host_pybind"):
        processed_section = transform_host_pybind_source(processed_section)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.writelines(processed_section)
    print(f"Created: {output_file}")

    created_files = [current_filename]
    if current_filename.startswith("host_pybind"):
        host_ttnn_filename = make_host_ttnn_filename(current_filename)
        if host_ttnn_filename:
            try:
                host_ttnn_source = generate_host_ttnn_source(
                    processed_section, current_filename
                )
                host_ttnn_file = output_path / host_ttnn_filename
                with open(host_ttnn_file, 'w', encoding='utf-8') as out_f:
                    out_f.writelines(host_ttnn_source)
                print(f"Created: {host_ttnn_file}")
                created_files.append(host_ttnn_filename)
            except Exception as ex:
                print(
                    f"Warning: failed to generate {host_ttnn_filename} from {current_filename}: {ex}"
                )

    return created_files


def extract_function_name(comment_line):
    """
    Extract a sanitized function name from a comment line.
    
    Args:
        comment_line: A comment line like "// matmul_kernel__d0i0_d1i0__f01__c0mem_c1mem__compute"
    
    Returns:
        A sanitized filename (e.g., "matmul_kernel__d0i0_d1i0__f01__c0mem_c1mem__compute.cpp")
    """
    # Remove "// " prefix and strip whitespace
    name = comment_line.strip()
    if name.startswith("//"):
        name = name[2:].strip()
    
    # Replace any characters that might be problematic in filenames
    # Keep alphanumeric, underscores, and hyphens
    name = re.sub(r'[^\w\-]', '_', name)
    
    # Keep the output focused on the last semantic part of the function name.
    # Example: batch_mm_accept__compute -> compute.cpp
    if "__" in name:
        name = name.rsplit("__", 1)[-1]
    elif "_" in name:
        name = name.rsplit("_", 1)[-1]

    return f"{name}.cpp"


def split_kernel_file(input_file, output_dir):
    """
    Split a kernel.cpp file into separate files based on comment markers.
    
    Args:
        input_file: Path to the input kernel.cpp file
        output_dir: Directory where output files will be written
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Pattern to match function marker comments emitted by translation.
    # Examples:
    #   // matmul__...__compute
    #   // batch_mm_accept__reader
    function_marker_pattern = re.compile(
        r'^\s*//\s*[A-Za-z_]\w*(?:__\w+)+\s*$'
    )
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_section = []
    current_filename = None
    section_count = 0
    used_filenames = set()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line is a function marker comment
        if function_marker_pattern.match(line):
            # Save previous section if it exists
            if current_filename and current_section:
                for created_name in _write_section_outputs(
                    output_path, current_filename, current_section
                ):
                    used_filenames.add(created_name)
            
            # Start new section
            current_section = [line]
            current_filename = extract_function_name(line)
            # Avoid accidental overwrite when multiple sections resolve to
            # the same suffix-based filename.
            if current_filename in used_filenames:
                stem = Path(current_filename).stem
                ext = Path(current_filename).suffix
                suffix = 2
                candidate = f"{stem}_{suffix}{ext}"
                while candidate in used_filenames:
                    suffix += 1
                    candidate = f"{stem}_{suffix}{ext}"
                current_filename = candidate
            used_filenames.add(current_filename)
            section_count += 1
        else:
            # Add line to current section
            if current_section is not None:
                current_section.append(line)
            else:
                # If we haven't encountered a marker yet, this might be a header section
                # We'll include it in the first section when we find one
                pass
        
        i += 1
    
    # Save the last section
    if current_filename and current_section:
        for created_name in _write_section_outputs(
            output_path, current_filename, current_section
        ):
            used_filenames.add(created_name)

    # The host split now emits host_cpp.cpp and host_pybind.cpp. Remove a stale
    # legacy host.cpp if it was left behind by an older run so callers don't
    # accidentally pick up the wrong artifact.
    legacy_host = output_path / "host.cpp"
    if (
        legacy_host.exists()
        and "host.cpp" not in used_filenames
        and (
            "host_cpp.cpp" in used_filenames
            or "host_pybind.cpp" in used_filenames
        )
    ):
        legacy_host.unlink()
        print(f"Removed stale: {legacy_host}")
    
    print(f"\nSplit complete: {section_count} function(s) extracted to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Split kernel.cpp into separate files based on comment markers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_kernel.py kernel.cpp -o output/
  python split_kernel.py kernel.cpp --output-dir kernels/
        """
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input kernel.cpp file'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='/root/tt-metal/tt_metal/programming_examples/mlir_matmul_simple/kernels',
        help='Output directory for split files (default: split_output)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return 1
    
    split_kernel_file(args.input_file, args.output_dir)
    return 0


if __name__ == '__main__':
    exit(main())
