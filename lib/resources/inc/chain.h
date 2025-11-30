#pragma once

#include "resource_base.h"
#include <string>

namespace scaleout {
namespace resources {

/**
 * @brief Chain interconnect resource representing daisy-chain wires/routers
 *
 * Models a simple chain interconnect among cores/tiles. It is created without
 * parameters and can only be consumed as a whole.
 */
class Chain : public Resource<Chain> {
private:
  bool is_available_;

public:
  explicit Chain(const std::string &resource_name = "")
      : Resource<Chain>(resource_name), is_available_(true) {
    initializeResourceName();
  }

  std::string getResourceTypeName() const override { return "Chain"; }
  bool isAvailable() const override { return is_available_; }
  void reset() override { is_available_ = true; }

  // Consumption model: all-or-nothing
  bool consume() {
    if (!is_available_) {
      return false;
    }
    is_available_ = false;
    return true;
  }

  void release() { is_available_ = true; }
};

} // namespace resources
} // namespace scaleout
