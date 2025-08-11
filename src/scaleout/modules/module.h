#pragma once

#include <string>
#include <vector>

namespace scaleout {
namespace modules {

/**
 * Base class for hardware modules built from primitive resources.
 *
 * A Module represents a topological arrangement of cores and the
 * interconnect/memory resources they require. Concrete modules are
 * responsible for checking resource availability and consuming/releasing
 * them when an operation is mapped.
 */
class Module {
private:
  std::string module_name_;
  std::vector<int> core_ids_;

public:
  explicit Module(std::string module_name, std::vector<int> core_ids)
      : module_name_(std::move(module_name)), core_ids_(std::move(core_ids)) {}

  virtual ~Module() = default;

  const std::string &getName() const { return module_name_; }

  const std::vector<int> &getCoreIds() const { return core_ids_; }

  /** Return a short, human-readable type name (e.g., "Torus", "Mesh2D"). */
  virtual std::string getTypeName() const = 0;
};

} // namespace modules
} // namespace scaleout
