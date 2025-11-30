#include "../inc/memory.h"

namespace scaleout {
namespace resources {

// MemoryCapacity
MemoryCapacity::MemoryCapacity(size_t total_size,
                               const std::string &resource_name)
    : Resource<MemoryCapacity>(resource_name), total_size_(total_size),
      available_size_(total_size) {
  initializeResourceName();
}

double MemoryCapacity::getUtilizationPercentage() const {
  if (total_size_ == 0) {
    return 0.0;
  }
  return static_cast<double>(getUsedSize()) / static_cast<double>(total_size_) *
         100.0;
}

bool MemoryCapacity::consume(size_t size) {
  if (size > available_size_) {
    return false;
  }
  available_size_ -= size;
  return true;
}

bool MemoryCapacity::release(size_t size) {
  if (size > (total_size_ - available_size_)) {
    return false;
  }
  available_size_ = std::min(total_size_, available_size_ + size);
  return true;
}

bool MemoryCapacity::canConsume(size_t size) const {
  return size <= available_size_;
}

void MemoryCapacity::reset() { available_size_ = total_size_; }

// MemoryPort
MemoryPort::MemoryPort(PortType port_type, size_t port_width,
                       const std::string &resource_name)
    : Resource<MemoryPort>(resource_name), port_type_(port_type),
      port_width_(port_width), is_available_(true) {
  initializeResourceName();
}

bool MemoryPort::acquire() {
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
