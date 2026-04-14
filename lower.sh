#!/bin/bash

set -e

TILELOOM_PASS_ARG="--loom-tileloom-to-ttkernel"
if [[ -n "${TILELOOM_TO_TTKERNEL_OPTIONS:-}" ]]; then
  TILELOOM_PASS_ARG="--loom-tileloom-to-ttkernel=${TILELOOM_TO_TTKERNEL_OPTIONS}"
fi

#/root/loom-dataflow/build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt --loom-tileloom-to-ttkernel  --mlir-print-ir-after-failure /root/loom-dataflow/test/Passes/flashattn_2Dmesh/IR/test.mlir -o kernel_ttkernel.mlir -o kernel_ttkernel.mlir


#/root/loom-dataflow/build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt "${TILELOOM_PASS_ARG}" /root/loom-dataflow/test/Passes/mm_2Dmesh/test2.mlir -o kernel_ttkernel.mlir

/root/loom-dataflow/build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt "${TILELOOM_PASS_ARG}" /root/loom-dataflow/test/Passes/mamba_chunk_scan/test.mlir -o kernel_ttkernel.mlir

python3 replace.py kernel_ttkernel.mlir

/root/tt-mlir/build/bin/ttmlir-opt --convert-ttkernel-to-emitc -canonicalize -cse -canonicalize -sccp -canonicalize kernel_ttkernel.mlir -o kernel_emitc.mlir

# Fold EmitC SSA temporaries (especially cast chains like i32<->ui32/ptrdiff_t)
# into expressions so generated C++ is substantially cleaner.
/root/tt-mlir/build/bin/ttmlir-opt --form-expressions kernel_emitc.mlir -o kernel_emitc_formexpr.mlir
mv kernel_emitc_formexpr.mlir kernel_emitc.mlir

/root/loom-dataflow/build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt --loom-post-emitc-host-signature kernel_emitc.mlir -o kernel_emitc_hostsig.mlir

/root/tt-mlir/build/bin/ttmlir-translate --ttkernel-to-cpp kernel_emitc_hostsig.mlir -o kernel.cpp

#python3 split_kernel.py kernel.cpp -o /tt-metal/ttnn/cpp/ttnn/operations/transformer/mlir_sdpa/device/kernels/
python3 split_kernel.py kernel.cpp
