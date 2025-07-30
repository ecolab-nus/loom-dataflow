#!/bin/bash

# IDE Setup Script for TMD C++ Project
# This script configures the development environment for better IDE support

set -e

echo "🔧 Setting up IDE configuration for C++ project..."

# Clean and rebuild to generate compile_commands.json
echo "📦 Cleaning previous build..."
rm -rf build

echo "🏗️  Building project with debug symbols..."
mkdir -p build
cd build

# Configure with debug symbols and compile commands export
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

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