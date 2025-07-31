#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace scaleout {
namespace resources {

/**
 * @brief Manages all resource instances across different resource types
 *
 * The ResourceManager provides centralized management for all resources,
 * including creation, tracking, and lifecycle management.
 */
class ResourceManager {
private:
  // Base class for type-erased resource storage
  struct ResourceContainer {
    virtual ~ResourceContainer() = default;
    virtual std::string getTypeName() const = 0;
    virtual size_t getCount() const = 0;
    virtual std::vector<uint64_t> getAllIds() const = 0;
    virtual bool removeById(uint64_t id) = 0;
    virtual void clear() = 0;
  };

  // Templated container for specific resource types
  template <typename T>
  struct TypedResourceContainer : public ResourceContainer {
    std::vector<std::shared_ptr<T>> resources;
    mutable std::mutex mutex;

    std::string getTypeName() const override {
      if (!resources.empty()) {
        return resources[0]->getResourceTypeName();
      }
      return typeid(T).name();
    }

    size_t getCount() const override {
      std::lock_guard<std::mutex> lock(mutex);
      return resources.size();
    }

    std::vector<uint64_t> getAllIds() const override {
      std::lock_guard<std::mutex> lock(mutex);
      std::vector<uint64_t> ids;
      for (const auto &resource : resources) {
        ids.push_back(resource->getResourceId());
      }
      return ids;
    }

    bool removeById(uint64_t id) override {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = std::remove_if(resources.begin(), resources.end(),
                               [id](const std::shared_ptr<T> &resource) {
                                 return resource->getResourceId() == id;
                               });
      if (it != resources.end()) {
        resources.erase(it, resources.end());
        return true;
      }
      return false;
    }

    void clear() override {
      std::lock_guard<std::mutex> lock(mutex);
      resources.clear();
    }

    void addResource(std::shared_ptr<T> resource) {
      std::lock_guard<std::mutex> lock(mutex);
      resources.push_back(resource);
    }

    std::shared_ptr<T> findById(uint64_t id) const {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = std::find_if(resources.begin(), resources.end(),
                             [id](const std::shared_ptr<T> &resource) {
                               return resource->getResourceId() == id;
                             });
      return (it != resources.end()) ? *it : nullptr;
    }

    std::vector<std::shared_ptr<T>> getAllResources() const {
      std::lock_guard<std::mutex> lock(mutex);
      return resources;
    }
  };

  std::unordered_map<std::type_index, std::unique_ptr<ResourceContainer>>
      containers_;
  mutable std::mutex manager_mutex_;

  template <typename T> TypedResourceContainer<T> *getContainer() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    auto type_index = std::type_index(typeid(T));
    auto it = containers_.find(type_index);

    if (it == containers_.end()) {
      auto container = std::make_unique<TypedResourceContainer<T>>();
      auto *ptr = container.get();
      containers_[type_index] = std::move(container);
      return ptr;
    }

    return static_cast<TypedResourceContainer<T> *>(it->second.get());
  }

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
   * @brief Create a new resource instance
   * @tparam T Resource type to create
   * @tparam Args Constructor argument types
   * @param args Constructor arguments
   * @return std::shared_ptr<T> Shared pointer to the created resource
   */
  template <typename T, typename... Args>
  std::shared_ptr<T> createResource(Args &&...args) {
    auto resource = std::make_shared<T>(std::forward<Args>(args)...);
    auto *container = getContainer<T>();
    container->addResource(resource);
    return resource;
  }

  /**
   * @brief Find a resource by its unique ID
   * @tparam T Resource type to find
   * @param id Unique resource ID
   * @return std::shared_ptr<T> Shared pointer to the resource, or nullptr if
   * not found
   */
  template <typename T> std::shared_ptr<T> findResource(uint64_t id) const {
    auto *container = const_cast<ResourceManager *>(this)->getContainer<T>();
    return container->findById(id);
  }

  /**
   * @brief Get all resources of a specific type
   * @tparam T Resource type to retrieve
   * @return std::vector<std::shared_ptr<T>> Vector of all resources of type T
   */
  template <typename T>
  std::vector<std::shared_ptr<T>> getAllResources() const {
    auto *container = const_cast<ResourceManager *>(this)->getContainer<T>();
    return container->getAllResources();
  }

  /**
   * @brief Get count of resources of a specific type
   * @tparam T Resource type to count
   * @return size_t Number of resources of type T
   */
  template <typename T> size_t getResourceCount() const {
    auto *container = const_cast<ResourceManager *>(this)->getContainer<T>();
    return container->getCount();
  }

  /**
   * @brief Remove a resource by its ID
   * @tparam T Resource type
   * @param id Unique resource ID
   * @return true if resource was found and removed, false otherwise
   */
  template <typename T> bool removeResource(uint64_t id) {
    auto *container = getContainer<T>();
    return container->removeById(id);
  }

  /**
   * @brief Clear all resources of a specific type
   * @tparam T Resource type to clear
   */
  template <typename T> void clearResources() {
    auto *container = getContainer<T>();
    container->clear();
  }

  /**
   * @brief Clear all resources of all types
   */
  void clearAllResources() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto &pair : containers_) {
      pair.second->clear();
    }
  }

  /**
   * @brief Get statistics for all managed resource types
   * @return std::unordered_map<std::string, size_t> Map of type names to counts
   */
  std::unordered_map<std::string, size_t> getResourceStatistics() const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    std::unordered_map<std::string, size_t> stats;
    for (const auto &pair : containers_) {
      stats[pair.second->getTypeName()] = pair.second->getCount();
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