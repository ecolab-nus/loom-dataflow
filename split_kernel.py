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


def _normalize_token(token):
    tok = token.strip()
    if tok.endswith(","):
        tok = tok[:-1].strip()
    return tok


def _parse_cb_or_int(token, semaphore_id_map=None):
    tok = _normalize_token(token)
    cb_match = re.search(r"CBIndex::c_(\d+)", tok)
    if cb_match:
        return int(cb_match.group(1))
    if re.fullmatch(r"\d+", tok):
        return int(tok)
    if semaphore_id_map and tok in semaphore_id_map:
        return int(semaphore_id_map[tok])
    raise ValueError(f"unsupported runtime token: {token}")


def _parse_core_coord_token(token):
    tok = _normalize_token(token)
    if tok == "core.x":
        return "core_x"
    if tok == "core.y":
        return "core_y"
    if re.fullmatch(r"\d+", tok):
        return int(tok)
    raise ValueError(f"unsupported core coord token: {token}")


def _is_core_coord_token(token):
    try:
        _parse_core_coord_token(token)
        return True
    except ValueError:
        return False


def _extract_host_ttnn_metadata(lines):
    if not lines:
        raise ValueError("empty host_pybind section")

    wrapper_base = extract_marker_symbol(lines[0])
    if wrapper_base.endswith("__host_pybind"):
        wrapper_base = wrapper_base[: -len("__host_pybind")]
    wrapper_name = re.sub(r"[^\w]", "_", wrapper_base)
    if not wrapper_name:
        wrapper_name = "generated_kernel"
    if wrapper_name[0].isdigit():
        wrapper_name = f"k_{wrapper_name}"

    buffer_bindings = []
    for line in lines:
        match = re.search(r"auto\*\s+(\w+)\s*=\s*v(\d+)\.buffer\(\);", line)
        if not match:
            continue
        buffer_name = match.group(1)
        arg_index = int(match.group(2)) - 1
        buffer_bindings.append((arg_index, buffer_name))

    if not buffer_bindings:
        raise ValueError("no tensor buffer bindings found")
    buffer_bindings.sort(key=lambda item: item[0])

    param_order = []
    param_roles = {}
    buffer_to_param = {}
    for _, buffer_name in buffer_bindings:
        param = buffer_name
        if param.endswith("_dram_buffer"):
            param = param[: -len("_dram_buffer")]
        buffer_to_param[buffer_name] = param
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
    for line in lines:
        match = re.search(r"auto\s+(\w+)\s*=\s*.*CreateSemaphore\(", line)
        if match:
            semaphore_vars.append(match.group(1))
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
        for cb_index_text in re.findall(r"CBIndex::c_(\d+)", cb_entries):
            cb_index = int(cb_index_text)
            if cb_index in seen_cb_indices:
                continue
            seen_cb_indices.add(cb_index)
            cb_layouts.append(
                {
                    "cb_index": cb_index,
                    "total_size": total_size,
                    "page_size": symbol_values["single_tile_size"],
                }
            )
    cb_layouts.sort(key=lambda item: item["cb_index"])

    compile_arg_order = []
    for line in lines:
        match = re.search(r"TensorAccessorArgs\((\w+)\)\.append_to\(compile_args\);", line)
        if not match:
            continue
        buffer_name = match.group(1)
        if buffer_name not in buffer_to_param:
            continue
        compile_arg_order.append(buffer_to_param[buffer_name])

    kernel_sources = {}
    for line in lines:
        match = re.search(
            r'auto\s+(\w+)\s*=\s*.*CreateKernel\(program,\s*OVERRIDE_KERNEL_PREFIX\s*"([^"]+)"',
            line,
        )
        if not match:
            continue
        kernel_id_var = match.group(1)
        kernel_sources[kernel_id_var] = match.group(2)

    reader_kernel = kernel_sources.get("reader_id")
    writer_kernel = kernel_sources.get("writer_id")
    compute_kernel = kernel_sources.get("compute_kernel_id")
    if not reader_kernel or not writer_kernel or not compute_kernel:
        raise ValueError("missing one or more kernel source paths")

    runtime_start = None
    runtime_end = None
    for idx, line in enumerate(lines):
        if "std::vector<uint32_t> runtime_args_for_core = {" in line:
            runtime_start = idx + 1
            continue
        if runtime_start is not None and line.strip() == "};":
            runtime_end = idx
            break

    if runtime_start is None or runtime_end is None:
        raise ValueError("runtime args block not found")

    runtime_tokens = []
    for raw_line in lines[runtime_start:runtime_end]:
        token = _normalize_token(raw_line)
        if token:
            runtime_tokens.append(token)

    memref_count = len(param_order)
    runtime_prefix_count = memref_count * 11
    if len(runtime_tokens) < runtime_prefix_count + 2:
        raise ValueError("runtime args block too short")

    runtime_memrefs = []
    for memref_idx in range(memref_count):
        base = memref_idx * 11
        cb_token = runtime_tokens[base]
        addr_token = runtime_tokens[base + 1]
        mcast_marker = runtime_tokens[base + 2]
        sender_sem = runtime_tokens[base + 9]
        receiver_sem = runtime_tokens[base + 10]

        cb_index = _parse_cb_or_int(cb_token)
        addr_match = re.search(r"(\w+)->address\(\)", addr_token)
        if not addr_match:
            raise ValueError(f"unsupported address token: {addr_token}")
        buffer_name = addr_match.group(1)
        if buffer_name not in buffer_to_param:
            raise ValueError(f"unknown buffer in runtime args: {buffer_name}")

        if mcast_marker.startswith("horizontal_"):
            mcast_kind = "horizontal"
        elif mcast_marker.startswith("vertical_"):
            mcast_kind = "vertical"
        elif mcast_marker.startswith("all_"):
            mcast_kind = "all"
        else:
            mcast_kind = "none"

        if sender_sem == "0":
            sender_sem = None
        if receiver_sem == "0":
            receiver_sem = None

        runtime_memrefs.append(
            {
                "tensor": buffer_to_param[buffer_name],
                "cb_index": cb_index,
                "mcast_kind": mcast_kind,
                "sender_sem": sender_sem,
                "receiver_sem": receiver_sem,
            }
        )

    # CompileArgTracker materializes core coordinates as two consecutive runtime
    # args. Locate that pair and treat any tokens before/after it as additional
    # runtime args that must keep relative order.
    core_coord_start = None
    for idx in range(runtime_prefix_count, len(runtime_tokens) - 1):
        if _is_core_coord_token(runtime_tokens[idx]) and _is_core_coord_token(
            runtime_tokens[idx + 1]
        ):
            core_coord_start = idx
            break
    if core_coord_start is None:
        raise ValueError("failed to locate core coordinate runtime args")

    runtime_pre_core = [
        _parse_cb_or_int(token, semaphore_id_map=semaphore_id_map)
        for token in runtime_tokens[runtime_prefix_count:core_coord_start]
    ]
    core_coord_order = [
        _parse_core_coord_token(runtime_tokens[core_coord_start]),
        _parse_core_coord_token(runtime_tokens[core_coord_start + 1]),
    ]
    runtime_post_core = [
        _parse_cb_or_int(token, semaphore_id_map=semaphore_id_map)
        for token in runtime_tokens[core_coord_start + 2 :]
    ]

    return {
        "wrapper_name": wrapper_name,
        "param_order": param_order,
        "input_param_order": input_param_order,
        "output_param_order": output_param_order,
        "compile_arg_order": compile_arg_order,
        "cb_layouts": cb_layouts,
        "semaphore_id_map": semaphore_id_map,
        "runtime_memrefs": runtime_memrefs,
        "runtime_pre_core": runtime_pre_core,
        "core_coord_order": core_coord_order,
        "runtime_post_core": runtime_post_core,
        "reader_kernel": reader_kernel,
        "writer_kernel": writer_kernel,
        "compute_kernel": compute_kernel,
    }


def generate_host_ttnn_source(host_pybind_lines, source_cpp_filename):
    metadata = _extract_host_ttnn_metadata(host_pybind_lines)
    wrapper_fn = f"{metadata['wrapper_name']}_ttnn"
    param_sig = ", ".join(
        metadata["param_order"]
        + [
            "start_core_x=UINT32_MAX",
            "start_core_y=UINT32_MAX",
            "end_core_x=UINT32_MAX",
            "end_core_y=UINT32_MAX",
        ]
    )

    source = []
    source.append("# Auto-generated by split_kernel.py from host_pybind.cpp.\n")
    source.append(f"# Source section: {source_cpp_filename}\n")
    source.append("import ttnn\n\n")
    source.append("UINT32_MAX = (1 << 32) - 1\n")
    source.append("SINGLE_TILE_SIZE = 2 * 1024\n\n")
    source.append(f"_PARAM_ORDER = {json.dumps(metadata['param_order'])}\n")
    source.append(f"_INPUT_PARAM_ORDER = {json.dumps(metadata['input_param_order'])}\n")
    source.append(f"_OUTPUT_PARAM_ORDER = {json.dumps(metadata['output_param_order'])}\n")
    source.append(f"_COMPILE_ARG_ORDER = {json.dumps(metadata['compile_arg_order'])}\n")
    source.append(f"_CB_LAYOUTS = {repr(metadata['cb_layouts'])}\n")
    source.append(f"_SEMAPHORE_IDS = {repr(metadata['semaphore_id_map'])}\n")
    source.append("SEMAPHORE_COUNT = len(_SEMAPHORE_IDS)\n")
    source.append(f"_RUNTIME_MEMREFS = {repr(metadata['runtime_memrefs'])}\n")
    source.append(f"_RUNTIME_PRE_CORE = {repr(metadata['runtime_pre_core'])}\n")
    source.append(f"_CORE_COORD_ORDER = {repr(metadata['core_coord_order'])}\n")
    source.append(f"_RUNTIME_POST_CORE = {repr(metadata['runtime_post_core'])}\n")
    source.append(
        f"_KERNEL_SOURCES = {repr({'reader': metadata['reader_kernel'], 'writer': metadata['writer_kernel'], 'compute': metadata['compute_kernel']})}\n\n"
    )
    source.append("def _logical_to_worker_core(device, core_x, core_y):\n")
    source.append("    logical_core = ttnn.CoreCoord(int(core_x), int(core_y))\n")
    source.append("    if hasattr(device, \"worker_core_from_logical_core\"):\n")
    source.append("        physical = device.worker_core_from_logical_core(logical_core)\n")
    source.append("        return int(physical.x), int(physical.y)\n")
    source.append("    return int(logical_core.x), int(logical_core.y)\n\n")
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
    source.append("    start_core_x = 0 if start_core_x == UINT32_MAX else int(start_core_x)\n")
    source.append("    start_core_y = 0 if start_core_y == UINT32_MAX else int(start_core_y)\n")
    source.append(
        "    end_core_x = int(core_grid.x - 1) if end_core_x == UINT32_MAX else int(end_core_x)\n"
    )
    source.append(
        "    end_core_y = int(core_grid.y - 1) if end_core_y == UINT32_MAX else int(end_core_y)\n"
    )
    source.append("    if start_core_x != 0 or start_core_y != 0:\n")
    source.append(
        "        raise ValueError(\"host_ttnn.py currently requires start_core_x/start_core_y to be 0\")\n"
    )
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
    source.append("        cb_format = ttnn.CBFormatDescriptor(\n")
    source.append("            buffer_index=int(cb_layout[\"cb_index\"]),\n")
    source.append("            data_format=ttnn.bfloat16,\n")
    source.append("            page_size=int(cb_layout[\"page_size\"]),\n")
    source.append("        )\n")
    source.append("        cbs.append(\n")
    source.append("            ttnn.CBDescriptor(\n")
    source.append("                total_size=int(cb_layout[\"total_size\"]),\n")
    source.append("                core_ranges=core_ranges,\n")
    source.append("                format_descriptors=[cb_format],\n")
    source.append("            )\n")
    source.append("        )\n")
    source.append("\n")
    source.append(
        "    semaphores = [ttnn.SemaphoreDescriptor(core_ranges=core_ranges, initial_value=UINT32_MAX) for _ in range(SEMAPHORE_COUNT)]\n"
    )
    source.append("\n")
    source.append("    compile_args = []\n")
    source.append("    for param_name in _COMPILE_ARG_ORDER:\n")
    source.append(
        "        compile_args.extend(ttnn.TensorAccessorArgs(param_bindings[param_name]).get_compile_time_args())\n"
    )
    source.append("\n")
    source.append("    runtime_args = [\n")
    source.append("        [[] for _ in range(end_core_y + 1)]\n")
    source.append("        for _ in range(end_core_x + 1)\n")
    source.append("    ]\n")
    source.append("    num_cores_with_work_c = end_core_x - start_core_x + 1\n")
    source.append("    num_cores_with_work_r = end_core_y - start_core_y + 1\n")
    source.append("    for core_x in range(start_core_x, end_core_x + 1):\n")
    source.append("        for core_y in range(start_core_y, end_core_y + 1):\n")
    source.append("            horizontal_sender_noc_x, horizontal_sender_noc_y = _logical_to_worker_core(device, 0, core_y)\n")
    source.append(
        "            horizontal_dest_start_noc_x, horizontal_dest_start_noc_y = _logical_to_worker_core(device, max(1, start_core_x), core_y)\n"
    )
    source.append(
        "            horizontal_dest_end_noc_x, horizontal_dest_end_noc_y = _logical_to_worker_core(device, end_core_x, core_y)\n"
    )
    source.append("            vertical_sender_noc_x, vertical_sender_noc_y = _logical_to_worker_core(device, core_x, 0)\n")
    source.append(
        "            vertical_dest_start_noc_x, vertical_dest_start_noc_y = _logical_to_worker_core(device, core_x, max(1, start_core_y))\n"
    )
    source.append(
        "            vertical_dest_end_noc_x, vertical_dest_end_noc_y = _logical_to_worker_core(device, core_x, end_core_y)\n"
    )
    source.append("            all_sender_noc_x, all_sender_noc_y = _logical_to_worker_core(device, 0, 0)\n")
    source.append(
        "            all_dest_start_noc_x, all_dest_start_noc_y = _logical_to_worker_core(device, start_core_x, start_core_y)\n"
    )
    source.append(
        "            all_dest_end_noc_x, all_dest_end_noc_y = _logical_to_worker_core(device, end_core_x, end_core_y)\n"
    )
    source.append("            runtime_args_for_core = []\n")
    source.append("            for runtime_memref in _RUNTIME_MEMREFS:\n")
    source.append("                tensor = param_bindings[runtime_memref[\"tensor\"]]\n")
    source.append("                runtime_args_for_core.append(int(runtime_memref[\"cb_index\"]))\n")
    source.append("                runtime_args_for_core.append(int(tensor.buffer_address()))\n")
    source.append("                kind = runtime_memref[\"mcast_kind\"]\n")
    source.append("                if kind == \"horizontal\":\n")
    source.append("                    runtime_args_for_core.extend([\n")
    source.append("                        horizontal_dest_start_noc_x,\n")
    source.append("                        horizontal_dest_start_noc_y,\n")
    source.append("                        horizontal_dest_end_noc_x,\n")
    source.append("                        horizontal_dest_end_noc_y,\n")
    source.append("                        max(num_cores_with_work_c - 1, 0),\n")
    source.append("                        horizontal_sender_noc_x,\n")
    source.append("                        horizontal_sender_noc_y,\n")
    source.append("                    ])\n")
    source.append("                elif kind == \"vertical\":\n")
    source.append("                    runtime_args_for_core.extend([\n")
    source.append("                        vertical_dest_start_noc_x,\n")
    source.append("                        vertical_dest_start_noc_y,\n")
    source.append("                        vertical_dest_end_noc_x,\n")
    source.append("                        vertical_dest_end_noc_y,\n")
    source.append("                        max(num_cores_with_work_r - 1, 0),\n")
    source.append("                        vertical_sender_noc_x,\n")
    source.append("                        vertical_sender_noc_y,\n")
    source.append("                    ])\n")
    source.append("                elif kind == \"all\":\n")
    source.append("                    runtime_args_for_core.extend([\n")
    source.append("                        all_dest_start_noc_x,\n")
    source.append("                        all_dest_start_noc_y,\n")
    source.append("                        all_dest_end_noc_x,\n")
    source.append("                        all_dest_end_noc_y,\n")
    source.append(
        "                        max(num_cores_with_work_c * num_cores_with_work_r - 1, 0),\n"
    )
    source.append("                        all_sender_noc_x,\n")
    source.append("                        all_sender_noc_y,\n")
    source.append("                    ])\n")
    source.append("                else:\n")
    source.append("                    runtime_args_for_core.extend([0, 0, 0, 0, 0, 0, 0])\n")
    source.append("                sender_sem = runtime_memref[\"sender_sem\"]\n")
    source.append("                receiver_sem = runtime_memref[\"receiver_sem\"]\n")
    source.append("                if sender_sem is None or receiver_sem is None:\n")
    source.append("                    runtime_args_for_core.extend([0, 0])\n")
    source.append("                else:\n")
    source.append(
        "                    runtime_args_for_core.extend([int(_SEMAPHORE_IDS[sender_sem]), int(_SEMAPHORE_IDS[receiver_sem])])\n"
    )
    source.append("            runtime_args_for_core.extend(_RUNTIME_PRE_CORE)\n")
    source.append("            for coord_token in _CORE_COORD_ORDER:\n")
    source.append("                if coord_token == \"core_x\":\n")
    source.append("                    runtime_args_for_core.append(int(core_x))\n")
    source.append("                elif coord_token == \"core_y\":\n")
    source.append("                    runtime_args_for_core.append(int(core_y))\n")
    source.append("                else:\n")
    source.append("                    runtime_args_for_core.append(int(coord_token))\n")
    source.append("            runtime_args_for_core.extend(_RUNTIME_POST_CORE)\n")
    source.append("            runtime_args[core_x][core_y] = runtime_args_for_core\n")
    source.append("\n")
    source.append("    reader_kernel_descriptor = ttnn.KernelDescriptor(\n")
    source.append("        kernel_source=_KERNEL_SOURCES[\"reader\"],\n")
    source.append("        core_ranges=core_ranges,\n")
    source.append("        compile_time_args=list(compile_args),\n")
    source.append("        runtime_args=runtime_args,\n")
    source.append("        config=ttnn.ReaderConfigDescriptor(),\n")
    source.append("    )\n")
    source.append("    writer_kernel_descriptor = ttnn.KernelDescriptor(\n")
    source.append("        kernel_source=_KERNEL_SOURCES[\"writer\"],\n")
    source.append("        core_ranges=core_ranges,\n")
    source.append("        compile_time_args=list(compile_args),\n")
    source.append("        runtime_args=runtime_args,\n")
    source.append("        config=ttnn.WriterConfigDescriptor(),\n")
    source.append("    )\n")
    source.append("    compute_kernel_descriptor = ttnn.KernelDescriptor(\n")
    source.append("        kernel_source=_KERNEL_SOURCES[\"compute\"],\n")
    source.append("        core_ranges=core_ranges,\n")
    source.append("        compile_time_args=list(compile_args),\n")
    source.append("        runtime_args=runtime_args,\n")
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
