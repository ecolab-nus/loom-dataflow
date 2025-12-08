#!/bin/bash

# IDE Setup Script for LOOM C++ Project
# This script configures the development environment for better IDE support

set -e

echo "🔧 Setting up IDE configuration for C++ project..."

# Clean and rebuild to generate compile_commands.json
echo "📦 Cleaning previous build..."
rm -rf build

echo "🏗️  Building project with debug symbols..."
mkdir -p build
cd build

# MLIR directory (can be overridden by env)
MLIR_DIR=${MLIR_DIR:-/opt/llvm-mlir/lib/cmake/mlir}
# Prefer Python 'lit' on PATH; allow override via env
LLVM_EXTERNAL_LIT=${LLVM_EXTERNAL_LIT:-}

echo "Using MLIR_DIR=$MLIR_DIR"

# Auto-detect lit/llvm-lit if not provided (prefer 'lit')
if [[ -z "$LLVM_EXTERNAL_LIT" ]]; then
    if command -v lit >/dev/null 2>&1; then
        LLVM_EXTERNAL_LIT=$(command -v lit)
    elif command -v llvm-lit >/dev/null 2>&1; then
        LLVM_EXTERNAL_LIT=$(command -v llvm-lit)
    elif [[ -x "$HOME/llvm-project/build/bin/llvm-lit" ]]; then
        LLVM_EXTERNAL_LIT="$HOME/llvm-project/build/bin/llvm-lit"
    else
        echo "ERROR: lit/llvm-lit not found. Install one of:"
        # Removed apt python3-lit as it may not exist on Ubuntu
        echo "  - pipx install lit (preferred); then run: pipx ensurepath and open a new shell"
        echo "  - python3 -m venv ~/.venvs/lit && ~/.venvs/lit/bin/pip install lit"
        exit 1
    fi
fi

echo "Using LLVM_EXTERNAL_LIT=$LLVM_EXTERNAL_LIT"

# Configure with debug symbols and compile commands export
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DMLIR_DIR="$MLIR_DIR" -DLLVM_EXTERNAL_LIT="$LLVM_EXTERNAL_LIT" -DLLVM_USE_LINKER=lld

# Build the project
cmake --build . --parallel

# Verify compile_commands.json was created
if [ -f "compile_commands.json" ]; then
    echo "✅ compile_commands.json generated successfully"
    echo "   Location: $(pwd)/compile_commands.json"
else
    echo "❌ compile_commands.json was not generated"
    exit 1
fi

# Return to project root
cd ..

echo ""
echo "🎉 IDE setup complete!"
echo ""
echo "📋 What was configured:"
echo "   ✓ CMake export compile commands enabled"
echo "   ✓ VSCode C++ extension settings"
echo "   ✓ IntelliSense configuration"
echo "   ✓ Include paths for GoogleTest"
echo "   ✓ Build tasks and debug configurations"
echo "   ✓ Debug build with symbols generated"
echo ""
echo "🔄 Next steps:"
echo "   1. Restart your IDE (Cursor) to pick up the new configuration"
echo "   2. The C++ extension should now resolve symbols correctly"
echo "   3. Use Ctrl+Shift+P -> 'C/C++: Reload IntelliSense' if needed"
echo ""
echo "🚀 You can now use:"
echo "   • F5 to debug your application"
echo "   • Ctrl+Shift+P -> 'Tasks: Run Task' for build tasks"
echo "   • Full IntelliSense support with auto-completion"
echo ""