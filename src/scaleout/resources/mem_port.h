#pragma once

#include "resource_base.h"
#include <string>

namespace scaleout {
namespace resources {

/**
 * @brief MemoryPort resource class representing a memory port
 *
 * A memory port is a unique and indivisible resource that can handle
 * either read or write operations, characterized by its port width.
 */
class MemoryPort : public Resource<MemoryPort> {
public:
  enum class PortType { READ, WRITE, READ_WRITE };

private:
  PortType port_type_;
  size_t port_width_; // Port width in bits
  bool is_available_;

public:
  /**
   * @brief Construct a new Memory Port object
   *
   * @param port_type Type of the port (READ, WRITE, or READ_WRITE)
   * @param port_width Width of the port in bits
   * @param resource_name Optional name for this port resource
   */
  MemoryPort(PortType port_type, size_t port_width,
             const std::string &resource_name = "");

  /**
   * @brief Get the port type
   * @return PortType The type of this port
   */
  PortType getPortType() const { return port_type_; }

  /**
   * @brief Get the port width
   * @return size_t The width of this port in bits
   */
  size_t getPortWidth() const { return port_width_; }

  // Resource base class overrides
  std::string getResourceTypeName() const override { return "MemoryPort"; }
  bool isAvailable() const override { return is_available_; }
  void reset() override;

  /**
   * @brief Reserve this port for use
   * @return true if successfully reserved, false if already in use
   */
  bool reserve();

  /**
   * @brief Release this port, making it available again
   */
  void release();

  /**
   * @brief Check if this port can handle the specified operation
   * @param operation_type The type of operation (READ or WRITE)
   * @return true if the port supports this operation
   */
  bool supportsOperation(PortType operation_type) const;

  /**
   * @brief Get string representation of port type
   * @return std::string String representation of the port type
   */
  std::string getPortTypeString() const;
};

} // namespace resources
} // namespace scaleout
