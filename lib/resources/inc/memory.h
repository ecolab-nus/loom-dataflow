#pragma once

#include "resource_base.h"
#include <cstddef>
#include <string>

namespace scaleout {
namespace resources {

/**
 * MemoryCapacity resource representing on-chip SRAM capacity
 */
class MemoryCapacity : public Resource<MemoryCapacity> {
private:
  size_t total_size_;
  size_t available_size_;

public:
  MemoryCapacity(size_t total_size, const std::string &resource_name = "");

  size_t getTotalSize() const { return total_size_; }
  size_t getAvailableSize() const { return available_size_; }
  size_t getUsedSize() const { return total_size_ - available_size_; }

  std::string getResourceTypeName() const override { return "MemoryCapacity"; }
  bool isAvailable() const override { return !isFull(); }

  double getUtilizationPercentage() const;
  bool consume(size_t size);
  bool release(size_t size);
  bool canConsume(size_t size) const;
  void reset() override;

  bool isFull() const { return available_size_ == 0; }
  bool isEmpty() const { return available_size_ == total_size_; }
};

/**
 * MemoryPort resource representing an on-chip SRAM port
 */
class MemoryPort : public Resource<MemoryPort> {
public:
  enum class PortType { READ, WRITE, READ_WRITE };

private:
  PortType port_type_;
  size_t port_width_;
  bool is_available_;

public:
  MemoryPort(PortType port_type, size_t port_width,
             const std::string &resource_name = "");

  PortType getPortType() const { return port_type_; }
  size_t getPortWidth() const { return port_width_; }

  std::string getResourceTypeName() const override { return "MemoryPort"; }
  bool isAvailable() const override { return is_available_; }
  void reset() override;

  bool acquire();
  void release();
  bool supportsOperation(PortType operation_type) const;
  std::string getPortTypeString() const;
};

/**
 * MemoryBank: a convenience wrapper that groups memory capacity with
 * one read and one write port, enabling atomic acquire/release for
 * common memory transfers.
 */
class MemoryBank {
private:
  // Non-owning pointers. All three objects are owned by the containing
  // hardware model and must outlive this MemoryBank instance.
  MemoryCapacity *capacity_;
  MemoryPort *read_port_;
  MemoryPort *write_port_;

public:
  MemoryBank(MemoryCapacity *capacity, MemoryPort *read_port,
             MemoryPort *write_port)
      : capacity_(capacity), read_port_(read_port), write_port_(write_port) {}

  MemoryCapacity *getCapacity() const { return capacity_; }
  MemoryPort *getReadPort() const { return read_port_; }
  MemoryPort *getWritePort() const { return write_port_; }

  bool canAcquireForTransfer(size_t bytes) const {
    if (capacity_ != nullptr && !capacity_->canConsume(bytes)) {
      return false;
    }
    if (read_port_ != nullptr && !read_port_->isAvailable()) {
      return false;
    }
    if (write_port_ != nullptr && !write_port_->isAvailable()) {
      return false;
    }
    return true;
  }

  bool acquireForTransfer(size_t bytes) {
    if (!canAcquireForTransfer(bytes)) {
      return false;
    }
    bool read_acquired = true;
    bool write_acquired = true;
    if (read_port_ != nullptr) {
      read_acquired = read_port_->acquire();
      if (!read_acquired) {
        return false;
      }
    }
    if (write_port_ != nullptr) {
      write_acquired = write_port_->acquire();
      if (!write_acquired) {
        if (read_acquired && read_port_ != nullptr) {
          read_port_->release();
        }
        return false;
      }
    }
    if (capacity_ != nullptr) {
      if (!capacity_->consume(bytes)) {
        if (write_acquired && write_port_ != nullptr) {
          write_port_->release();
        }
        if (read_acquired && read_port_ != nullptr) {
          read_port_->release();
        }
        return false;
      }
    }
    return true;
  }

  void releaseTransfer(size_t bytes) {
    if (capacity_ != nullptr) {
      capacity_->release(bytes);
    }
    if (write_port_ != nullptr) {
      write_port_->release();
    }
    if (read_port_ != nullptr) {
      read_port_->release();
    }
  }
};

} // namespace resources
} // namespace scaleout
