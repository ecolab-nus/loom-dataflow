#pragma once

#include "resource_base.h"
#include <string>

namespace scaleout {
namespace resources {

/**
 * @brief Ring interconnect resource representing on-chip ring wires/routers
 *
 * Models a ring (or torus-style) on-chip network among cores/tiles. It is
 * created without parameters and can only be consumed as a whole.
 */
class Ring : public Resource<Ring> {
private:
  bool is_available_;

public:
  explicit Ring(const std::string &resource_name = "")
      : Resource<Ring>(resource_name), is_available_(true) {
    initializeResourceName();
  }

  std::string getResourceTypeName() const override { return "Ring"; }
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
