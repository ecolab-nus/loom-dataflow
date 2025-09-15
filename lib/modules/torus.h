#pragma once

#include "../resources/memory.h"
#include "../resources/ring.h"
#include "network.h"
#include <cstddef>
#include <vector>

namespace scaleout {
namespace modules {

/**
 * Torus module backed by a ring interconnect among a set of cores.
 *
 * Usage examples:
 * - Broadcast: consume ring once, and consume memory capacity per core
 *   for the broadcasted element(s).
 * - Reduction: consume ring once. No per-core memory reservation by default.
 */
class Torus : public NetworkModule {
private:
  resources::Ring &ring_;
  // Optional per-core local memory capacities; if provided, used for
  // broadcast accounting when no memory resource groups are provided.
  std::vector<resources::MemoryCapacity *> per_core_memory_;
  // Optional per-core memory banks (capacity + read/write ports)
  std::vector<resources::MemoryBank *> per_core_memory_groups_;

public:
  Torus(std::string moduleName, std::vector<int> core_ids,
        const mlir::AffineMap &coreIndexMap, resources::Ring &ring,
        std::vector<resources::MemoryCapacity *> per_core_memory = {},
        std::vector<resources::MemoryBank *> per_core_memory_groups = {})
      : NetworkModule(moduleName, std::move(core_ids), coreIndexMap),
        ring_(ring), per_core_memory_(std::move(per_core_memory)),
        per_core_memory_groups_(std::move(per_core_memory_groups)) {}

  std::string getTypeName() const override { return "Torus"; }

  // Returns true if the ring is free; does not mutate state.
  bool canBroadcast(size_t element_bytes) const {
    (void)element_bytes; // Currently unused except for mem checks
    if (!ring_.isAvailable()) {
      return false;
    }
    if (!per_core_memory_groups_.empty()) {
      for (const auto *group : per_core_memory_groups_) {
        if (group == nullptr || !group->canAcquireForTransfer(element_bytes)) {
          return false;
        }
      }
    } else if (!per_core_memory_.empty()) {
      for (const auto *mem : per_core_memory_) {
        if (mem == nullptr || !mem->canConsume(element_bytes)) {
          return false;
        }
      }
    }
    return true;
  }

  // Atomically acquire ring and deduct per-core memory for broadcast.
  bool acquireForBroadcast(size_t element_bytes) {
    if (!canBroadcast(element_bytes)) {
      return false;
    }
    if (!ring_.consume()) {
      return false;
    }
    if (!per_core_memory_groups_.empty()) {
      // Acquire ports + capacity per core
      for (auto *group : per_core_memory_groups_) {
        if (group == nullptr) {
          continue;
        }
        if (!group->acquireForTransfer(element_bytes)) {
          // Roll back all previously acquired groups and ring
          for (auto *rollback_group : per_core_memory_groups_) {
            if (rollback_group == group) {
              break;
            }
            if (rollback_group != nullptr) {
              rollback_group->releaseTransfer(element_bytes);
            }
          }
          ring_.release();
          return false;
        }
      }
    } else {
      // Capacity-only accounting
      for (auto *mem : per_core_memory_) {
        if (mem != nullptr) {
          bool ok = mem->consume(element_bytes);
          if (!ok) {
            // Roll back previous mem consumption and release ring
            for (auto *rollback_mem : per_core_memory_) {
              if (rollback_mem == mem) {
                break;
              }
              if (rollback_mem != nullptr) {
                rollback_mem->release(element_bytes);
              }
            }
            ring_.release();
            return false;
          }
        }
      }
    }
    return true;
  }

  // Release ring and credit back per-core memory for broadcast payload.
  void releaseBroadcast(size_t element_bytes) {
    if (!per_core_memory_groups_.empty()) {
      for (auto *group : per_core_memory_groups_) {
        if (group != nullptr) {
          group->releaseTransfer(element_bytes);
        }
      }
    } else {
      for (auto *mem : per_core_memory_) {
        if (mem != nullptr) {
          mem->release(element_bytes);
        }
      }
    }
    ring_.release();
  }

  // Reduction uses the ring; memory is not modified by default.
  bool acquireForReduction() { return ring_.consume(); }
  void releaseReduction() { ring_.release(); }
};

} // namespace modules
} // namespace scaleout
