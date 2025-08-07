#pragma once

#include "resource_base.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace scaleout {
namespace resources {

/**
 * @brief Simplified resource manager for managing all resource instances
 *
 * The ResourceManager provides centralized management for all resources,
 * including creation, tracking, and lifecycle management without type-based
 * organization.
 */
class ResourceManager {
private:
  std::vector<std::shared_ptr<ResourceBase>> resources_;
  mutable std::mutex manager_mutex_;

public:
  /**
   * @brief Get the singleton instance of ResourceManager
   * @return ResourceManager& The singleton instance
   */
  static ResourceManager &getInstance() {
    static ResourceManager instance;
    return instance;
  }

  /**
   * @brief Add a resource to the manager
   * @param resource Shared pointer to the resource to add
   */
  void addResource(std::shared_ptr<ResourceBase> resource) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    resources_.push_back(resource);
  }

  /**
   * @brief Find a resource by its unique ID
   * @param id Unique resource ID
   * @return std::shared_ptr<ResourceBase> Shared pointer to the resource, or
   * nullptr if not found
   */
  std::shared_ptr<ResourceBase> findResource(uint64_t id) const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    auto it = std::find_if(resources_.begin(), resources_.end(),
                           [id](const std::shared_ptr<ResourceBase> &resource) {
                             return resource->getResourceId() == id;
                           });
    return (it != resources_.end()) ? *it : nullptr;
  }

  /**
   * @brief Get all resources
   * @return std::vector<std::shared_ptr<ResourceBase>> Vector of all resources
   */
  std::vector<std::shared_ptr<ResourceBase>> getAllResources() const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    return resources_;
  }

  /**
   * @brief Get total count of resources
   * @return size_t Number of resources
   */
  size_t getResourceCount() const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    return resources_.size();
  }

  /**
   * @brief Remove a resource by its ID
   * @param id Unique resource ID
   * @return true if resource was found and removed, false otherwise
   */
  bool removeResource(uint64_t id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    auto it =
        std::remove_if(resources_.begin(), resources_.end(),
                       [id](const std::shared_ptr<ResourceBase> &resource) {
                         return resource->getResourceId() == id;
                       });
    if (it != resources_.end()) {
      resources_.erase(it, resources_.end());
      return true;
    }
    return false;
  }

  /**
   * @brief Clear all resources
   */
  void clearAllResources() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    resources_.clear();
  }

  /**
   * @brief Get statistics for all managed resource types
   * @return std::unordered_map<std::string, size_t> Map of type names to counts
   */
  std::unordered_map<std::string, size_t> getResourceStatistics() const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    std::unordered_map<std::string, size_t> stats;
    for (const auto &resource : resources_) {
      const std::string &typeName = resource->getResourceTypeName();
      stats[typeName]++;
    }
    return stats;
  }

private:
  ResourceManager() = default;
  ~ResourceManager() = default;
  ResourceManager(const ResourceManager &) = delete;
  ResourceManager &operator=(const ResourceManager &) = delete;
};

} // namespace resources
} // namespace scaleout