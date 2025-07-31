#include "mem_capacity.h"
#include <algorithm>

namespace scaleout {
namespace resources {

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

bool MemoryCapacity::allocate(size_t size) {
  if (size > available_size_) {
    return false;
  }
  available_size_ -= size;
  return true;
}

bool MemoryCapacity::deallocate(size_t size) {
  // Prevent overflow when deallocating
  if (size > (total_size_ - available_size_)) {
    return false;
  }

  available_size_ = std::min(total_size_, available_size_ + size);
  return true;
}

bool MemoryCapacity::canAllocate(size_t size) const {
  return size <= available_size_;
}

void MemoryCapacity::reset() { available_size_ = total_size_; }

} // namespace resources
} // namespace scaleout