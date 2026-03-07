#!/bin/bash

#/root/loom-dataflow/build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt --loom-tileloom-to-ttkernel  --mlir-print-ir-after-failure /root/loom-dataflow/test/Passes/flashattn_2Dmesh/IR/test.mlir -o kernel_ttkernel.mlir -o kernel_ttkernel.mlir


#/root/loom-dataflow/build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt --loom-tileloom-to-ttkernel /root/loom-dataflow/test/Passes/mm_2Dmesh/test2.mlir -o kernel_ttkernel.mlir

/root/loom-dataflow/build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt --loom-tileloom-to-ttkernel --mlir-print-ir-after-failure  /root/loom-dataflow/test/Passes/flashattn_2Dmesh/IR/test.mlir -o kernel_ttkernel.mlir

python3 replace.py kernel_ttkernel.mlir

/root/tt-mlir/build/bin/ttmlir-opt --convert-ttkernel-to-emitc -canonicalize -cse -canonicalize -sccp -canonicalize kernel_ttkernel.mlir -o kernel_emitc.mlir

/root/loom-dataflow/build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt --loom-post-emitc-host-signature kernel_emitc.mlir -o kernel_emitc_hostsig.mlir

/root/tt-mlir/build/bin/ttmlir-translate --ttkernel-to-cpp kernel_emitc_hostsig.mlir -o kernel.cpp

python3 split_kernel.py kernel.cpp
