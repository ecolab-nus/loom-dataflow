#pragma once

#include "resource_base.h"
#include <cstddef>
#include <string>

namespace scaleout {
namespace resources {

/**
 * @brief MemoryCapacity resource class representing memory capacity
 *
 * A memory capacity resource is characterized by its total size and
 * the amount of space currently available (size left).
 */
class MemoryCapacity : public Resource<MemoryCapacity> {
private:
  size_t total_size_;     // Total capacity in bytes
  size_t available_size_; // Available capacity in bytes

public:
  /**
   * @brief Construct a new Memory Capacity object
   *
   * @param total_size Total capacity in bytes
   * @param resource_name Optional name for this capacity resource
   */
  MemoryCapacity(size_t total_size, const std::string &resource_name = "");

  /**
   * @brief Get the total capacity
   * @return size_t Total capacity in bytes
   */
  size_t getTotalSize() const { return total_size_; }

  /**
   * @brief Get the available (remaining) capacity
   * @return size_t Available capacity in bytes
   */
  size_t getAvailableSize() const { return available_size_; }

  /**
   * @brief Get the used capacity
   * @return size_t Used capacity in bytes
   */
  size_t getUsedSize() const { return total_size_ - available_size_; }

  // Resource base class overrides
  std::string getResourceTypeName() const override { return "MemoryCapacity"; }
  bool isAvailable() const override { return !isFull(); }

  /**
   * @brief Get the utilization percentage
   * @return double Utilization as a percentage (0.0 to 100.0)
   */
  double getUtilizationPercentage() const;

  /**
   * @brief Allocate a specified amount of memory
   * @param size Amount to allocate in bytes
   * @return true if allocation successful, false if insufficient space
   */
  bool allocate(size_t size);

  /**
   * @brief Deallocate a specified amount of memory
   * @param size Amount to deallocate in bytes
   * @return true if deallocation successful, false if invalid size
   */
  bool deallocate(size_t size);

  /**
   * @brief Check if there's enough space for allocation
   * @param size Required size in bytes
   * @return true if space is available, false otherwise
   */
  bool canAllocate(size_t size) const;

  /**
   * @brief Reset the capacity to full availability
   */
  void reset() override;

  /**
   * @brief Check if the capacity is fully utilized
   * @return true if no space remaining, false otherwise
   */
  bool isFull() const { return available_size_ == 0; }

  /**
   * @brief Check if the capacity is completely free
   * @return true if no space is used, false otherwise
   */
  bool isEmpty() const { return available_size_ == total_size_; }
};

} // namespace resources
} // namespace scaleout