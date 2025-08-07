# An out-of-tree MLIR dialect

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with a standalone `opt`-like tool to operate on that dialect.

## Building LLVM first

download llvm-project and build it and install in your $HOME/opt/llvm-mlir
```sh
cd $LLVM_PROJECT/
mkdir build
cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
   -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
   -DCMAKE_INSTALL_PREFIX=$HOME/opt/llvm-mlir \
   -DLLVM_BUILD_UTILS=ON \
   -DLLVM_INSTALL_UTILS=ON
cmake --build . --target check-mlir
ninja install
```

## Building this

```sh
cmake -G Ninja .. \
      -DMLIR_DIR=$HOME/opt/llvm-mlir/lib/cmake/mlir \
      -DLLVM_EXTERNAL_LIT=$HOME/llvm-project/build/bin/llvm-lit \
      -DLLVM_USE_LINKER=lld

cmake --build . --target check-standalone
```

To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
