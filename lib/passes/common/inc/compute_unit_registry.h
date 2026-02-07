/**
 * @file compute_unit_registry.h
 * @brief Registry for binding compute operations to hardware units.
 * @details
 * This header provides a lightweight mechanism to bind MLIR compute operations
 * (e.g., linalg.matmul) to hardware units (e.g., FPU) with their throughput
 * and FLOP calculation coefficients. This enables extensible compute-bound
 * constraint generation without hardcoding operation types.
 */

#pragma once

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace loom {

/**
 * @brief Binding between a compute operation type and hardware unit.
 */
struct ComputeUnitBinding {
  /// @brief MLIR operation name string (e.g., "linalg.matmul")
  llvm::StringRef opName;

  /// @brief Hardware unit name from df.mat (e.g., "FPU")
  llvm::StringRef unitName;

  /// @brief Throughput from df.mat attribute
  int64_t throughput = 0;

  /// @brief Coefficient for FLOP calculation (2 for matmul FMA)
  int64_t flopCoefficient = 2;

  /// @brief Check if this binding matches the given operation
  bool matches(mlir::Operation *op) const {
    return op->getName().getStringRef() == opName;
  }
};

/**
 * @brief Registry for compute unit bindings.
 * @details
 * Maintains a list of bindings between compute operations and hardware units.
 * Can be extended at runtime to support new operation types.
 */
class ComputeUnitRegistry {
public:
  ComputeUnitRegistry() = default;

  /// @brief Register a new compute unit binding
  void registerBinding(ComputeUnitBinding binding) {
    bindings.push_back(std::move(binding));
  }

  /// @brief Find a binding for the given operation
  std::optional<ComputeUnitBinding> getBinding(mlir::Operation *op) const {
    for (const auto &binding : bindings) {
      if (binding.matches(op)) {
        return binding;
      }
    }
    return std::nullopt;
  }

  /// @brief Check if an operation has a registered compute unit
  bool hasBinding(mlir::Operation *op) const {
    return getBinding(op).has_value();
  }

  /// @brief Create a registry with default bindings (matmul -> FPU)
  static ComputeUnitRegistry createDefault(int64_t fpuThroughput) {
    ComputeUnitRegistry registry;

    // Register linalg.matmul -> FPU binding
    ComputeUnitBinding matmulBinding;
    matmulBinding.opName = "linalg.matmul";
    matmulBinding.unitName = "FPU";
    matmulBinding.throughput = fpuThroughput;
    matmulBinding.flopCoefficient = 2; // Multiply-accumulate

    registry.registerBinding(matmulBinding);

    return registry;
  }

private:
  llvm::SmallVector<ComputeUnitBinding, 4> bindings;
};

/**
 * @brief Hardware timing information for constraint generation.
 * @details
 * Aggregates all hardware parameters needed to compute the compute-memory
 * bound constraint: BW_B*sizeA + BW_A*sizeB - BW_A*BW_B*compute/T <= 0
 */
struct HardwareTiming {
  int64_t fpuThroughput = 0; ///< From df.mat "FPU" throughput attribute
  int64_t l1Bandwidth = 0;   ///< From df.memory "L1" bandwidth attribute
  int64_t dramBandwidth = 0; ///< From df.memory "DRAM" bandwidth attribute
  int64_t totalCores = 0;    ///< Product of spatial_dim sizes (e.g., 8*8=64)

  /// @brief Calculate effective bandwidth for a copy with given broadcast
  /// @param broadcastX Broadcast factor in X dimension
  /// @param broadcastY Broadcast factor in Y dimension
  /// @return min(L1_bandwidth, DRAM_bandwidth / competing_cores)
  int64_t getEffectiveBandwidth(int64_t broadcastX, int64_t broadcastY) const {
    int64_t broadcastMultiplier = broadcastX * broadcastY;
    int64_t competingCores = totalCores / broadcastMultiplier;
    if (competingCores <= 0)
      competingCores = 1;

    int64_t effectiveDramBw = dramBandwidth / competingCores;
    return std::min(l1Bandwidth, effectiveDramBw);
  }
};

} // namespace loom
