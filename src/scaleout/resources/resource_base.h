#pragma once

#include <atomic>
#include <string>

namespace scaleout {
namespace resources {

/**
 * @brief Simple base interface for all resources
 *
 * Provides common interface that all resources must implement.
 */
class ResourceBase {
public:
  virtual ~ResourceBase() = default;

  /**
   * @brief Get the unique resource ID
   * @return uint64_t The unique ID of this resource
   */
  virtual uint64_t getResourceId() const = 0;

  /**
   * @brief Get the resource name
   * @return const std::string& The name of this resource
   */
  virtual const std::string &getResourceName() const = 0;

  /**
   * @brief Set the resource name
   * @param name New name for this resource
   */
  virtual void setResourceName(const std::string &name) = 0;

  /**
   * @brief Get the type name of this resource
   * @return std::string The type name
   */
  virtual std::string getResourceTypeName() const = 0;

  /**
   * @brief Check if the resource is available for use
   * @return true if available, false otherwise
   */
  virtual bool isAvailable() const = 0;

  /**
   * @brief Get string representation of the resource
   * @return std::string String representation including ID and name
   */
  virtual std::string toString() const = 0;

  /**
   * @brief Reset the resource to its initial state
   */
  virtual void reset() = 0;
};

/**
 * @brief Base class template for all resources
 *
 * Provides unique ID generation and common resource interface.
 * All concrete resource types should inherit from this template.
 */
template <typename ResourceType> class Resource : public ResourceBase {
private:
  static std::atomic<uint64_t> next_id_;
  uint64_t resource_id_;
  std::string resource_name_;

protected:
  /**
   * @brief Generate a unique resource ID
   * @return uint64_t Unique ID for this resource instance
   */
  static uint64_t generateUniqueId() {
    return next_id_.fetch_add(1, std::memory_order_relaxed);
  }

  /**
   * @brief Initialize resource name if empty (called from derived constructors)
   */
  void initializeResourceName() {
    if (resource_name_.empty()) {
      resource_name_ =
          getResourceTypeName() + "_" + std::to_string(resource_id_);
    }
  }

public:
  /**
   * @brief Construct a new Resource object
   * @param resource_name Optional name for this resource instance
   */
  explicit Resource(const std::string &resource_name = "")
      : resource_id_(generateUniqueId()), resource_name_(resource_name) {
    // Note: resource_name_ will be initialized in derived class if empty
  }

  /**
   * @brief Virtual destructor for proper inheritance
   */
  virtual ~Resource() = default;

  /**
   * @brief Get the unique resource ID
   * @return uint64_t The unique ID of this resource
   */
  uint64_t getResourceId() const override { return resource_id_; }

  /**
   * @brief Get the resource name
   * @return const std::string& The name of this resource
   */
  const std::string &getResourceName() const override { return resource_name_; }

  /**
   * @brief Set the resource name
   * @param name New name for this resource
   */
  void setResourceName(const std::string &name) override {
    resource_name_ = name;
  }

  /**
   * @brief Get the type name of this resource
   * @return std::string The type name (to be implemented by derived classes)
   */
  virtual std::string getResourceTypeName() const override = 0;

  /**
   * @brief Check if the resource is available for use
   * @return true if available, false otherwise
   */
  virtual bool isAvailable() const override = 0;

  /**
   * @brief Get string representation of the resource
   * @return std::string String representation including ID and name
   */
  std::string toString() const override {
    return getResourceTypeName() + " [ID: " + std::to_string(resource_id_) +
           ", Name: " + resource_name_ + "]";
  }

  /**
   * @brief Reset the resource to its initial state
   */
  virtual void reset() override = 0;
};

// Static member definition
template <typename ResourceType>
std::atomic<uint64_t> Resource<ResourceType>::next_id_{1};

} // namespace resources
} // namespace scaleout