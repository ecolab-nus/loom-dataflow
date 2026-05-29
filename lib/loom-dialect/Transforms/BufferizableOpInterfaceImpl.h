#ifndef LOOM_DIALECT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
#define LOOM_DIALECT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;
class MLIRContext; // Added forward declaration for MLIRContext
} // namespace mlir

namespace loom {
void registerBufferizableOpInterfaceExternalModels(mlir::MLIRContext *ctx);
} // namespace loom

#endif // LOOM_DIALECT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
