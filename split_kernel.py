#!/usr/bin/env python3
"""
Script to split kernel.cpp into separate files based on comment markers.
Each function section (identified by a comment line like "// foo__compute")
will be written to its own file.
"""

import argparse
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
        for line in lines
    ]

    if section_name and section_name.startswith("host"):
        processed = insert_include_if_missing(processed, "#include <vector>\n")

    if section_name and section_name.startswith("compute"):
        processed = insert_include_if_missing(processed, '#include "math.h"\n')
        processed = insert_include_if_missing(processed, '#include "debug/dprint.h"\n')
        processed = insert_compute_trace_markers(processed)

    return processed


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
                output_file = output_path / current_filename
                processed_section = process_source_content(current_section, current_filename)
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.writelines(processed_section)
                print(f"Created: {output_file}")
            
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
        output_file = output_path / current_filename
        processed_section = process_source_content(current_section, current_filename)
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.writelines(processed_section)
        print(f"Created: {output_file}")
    
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
        default='/tt-metal/tt_metal/programming_examples/mlir_matmul_simple/kernels',
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
