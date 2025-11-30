#pragma once

#include <string>

namespace scaleout {
namespace modules {

/**
 * Base class for hardware modules built from primitive resources.
 *
 * A Module represents a way of using hardware resources (e.g., rings,
 * chains, memories). It does not model or own the set of cores linked by
 * a topology. Concrete modules are responsible for checking resource
 * availability and consuming/releasing them when an operation is mapped.
 */
class Module {
private:
  std::string module_name_;

public:
  explicit Module(std::string module_name)
      : module_name_(std::move(module_name)) {}

  virtual ~Module() = default;

  const std::string &getName() const { return module_name_; }

  /** Return a short, human-readable type name (e.g., "Torus", "Mesh2D"). */
  virtual std::string getTypeName() const = 0;
};

} // namespace modules
} // namespace scaleout
