#include "mem_port.h"

namespace scaleout {
namespace resources {

MemoryPort::MemoryPort(PortType port_type, size_t port_width,
                       const std::string &resource_name)
    : Resource<MemoryPort>(resource_name), port_type_(port_type),
      port_width_(port_width), is_available_(true) {
  initializeResourceName();
}

bool MemoryPort::reserve() {
  if (!is_available_) {
    return false;
  }
  is_available_ = false;
  return true;
}

void MemoryPort::release() { is_available_ = true; }

void MemoryPort::reset() { is_available_ = true; }

bool MemoryPort::supportsOperation(PortType operation_type) const {
  switch (port_type_) {
  case PortType::READ:
    return operation_type == PortType::READ;
  case PortType::WRITE:
    return operation_type == PortType::WRITE;
  case PortType::READ_WRITE:
    return operation_type == PortType::READ ||
           operation_type == PortType::WRITE;
  default:
    return false;
  }
}

std::string MemoryPort::getPortTypeString() const {
  switch (port_type_) {
  case PortType::READ:
    return "READ";
  case PortType::WRITE:
    return "WRITE";
  case PortType::READ_WRITE:
    return "READ_WRITE";
  default:
    return "UNKNOWN";
  }
}

} // namespace resources
} // namespace scaleout