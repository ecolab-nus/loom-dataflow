export LLVM_BINARY_DIR=<path-to-your-llvm-binaries>
export TRITON_SHARED_OPT_PATH=$TRITON_PLUGIN_DIRS/triton/build/<your-cmake-directory>/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export TRITON_SHARED_DUMP_PATH=/tmp/triton_ir
export TRITON_PLUGIN_DIRES=<path-to-triton-shared>
rm -rf ~/.triton/cache/*
# in case of using triton venv
source <path-to-triton-venv>/.venv/bin/activate
python3 <path-to-mm.py>