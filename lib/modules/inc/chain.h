#pragma once

#include "chain.h"  // resources::Chain (from resources/inc)
#include "network.h"

namespace scaleout {
namespace modules {

/**
 * Chain: 1D network module backed by a resources::Chain interconnect.
 * Inherits NetworkModule to carry core linkage and affine placement.
 */
class Chain : public NetworkModule {
private:
  resources::Chain &chain_;

public:
  Chain(std::string name, std::vector<int> core_ids,
        const mlir::AffineMap &coreIndexMap, resources::Chain &chain)
      : NetworkModule(name, std::move(core_ids), coreIndexMap), chain_(chain) {}

  std::string getTypeName() const override { return "Chain"; }

  bool isAvailable() const { return chain_.isAvailable(); }
  bool acquire() { return chain_.consume(); }
  void release() { chain_.release(); }
};

} // namespace modules
} // namespace scaleout
