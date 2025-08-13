#pragma once

#include "module.h"
#include <memory>
#include <string>
#include <vector>

// Unconditionally require MLIR AffineMap
#include "mlir/IR/AffineMap.h"

namespace scaleout {
namespace modules {

/**
 * NetworkModule links a set of cores and records their affine placement.
 *
 * This separates placement from resource-usage modules like Torus/Mesh2D.
 */
class NetworkModule : public Module {
private:
  // Participating core ids in physical order (implementation-defined).
  std::vector<int> core_ids_;

  struct Impl;
  std::unique_ptr<Impl> impl_;

public:
  // The affine map is mandatory.
  NetworkModule(std::string name, std::vector<int> core_ids,
                const mlir::AffineMap &coreIndexMap);
  ~NetworkModule();

  NetworkModule(const NetworkModule &) = delete;
  NetworkModule &operator=(const NetworkModule &) = delete;
  NetworkModule(NetworkModule &&) noexcept;
  NetworkModule &operator=(NetworkModule &&) noexcept;

  std::string getTypeName() const override { return "Network"; }

  const std::vector<int> &getCoreIds() const { return core_ids_; }

  // Access the MLIR AffineMap that maps logical indices to core ids.
  const mlir::AffineMap &getAffinePlacementMap() const;
};

} // namespace modules
} // namespace scaleout
