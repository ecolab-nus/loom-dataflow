#include "memory.h"
#include "resource_manager.h"
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

/**
 * @file resource_demo.cpp
 * @brief Dedicated executable for demonstrating the resource management system
 *
 * This program showcases the capabilities of the MemoryPort and MemoryCapacity
 * resource classes along with the ResourceManager system.
 */

/**
 * @brief Demonstrate basic resource creation and management
 */
void demonstrateBasicUsage() {
  std::cout << "=== Resource Management Example ===" << std::endl;

  // Get the resource manager instance
  auto &manager = scaleout::resources::ResourceManager::getInstance();

  // Create some on-chip SRAM ports
  auto read_port = std::make_shared<scaleout::resources::MemoryPort>(
      scaleout::resources::MemoryPort::PortType::READ, 64, "SRAM_Read_Port_0");
  auto write_port = std::make_shared<scaleout::resources::MemoryPort>(
      scaleout::resources::MemoryPort::PortType::WRITE, 64,
      "SRAM_Write_Port_0");
  auto rw_port = std::make_shared<scaleout::resources::MemoryPort>(
      scaleout::resources::MemoryPort::PortType::READ_WRITE, 128,
      "SRAM_RW_Port_0");

  // Create some on-chip SRAM capacities
  auto main_memory = std::make_shared<scaleout::resources::MemoryCapacity>(
      256 * 1024, "SPAD_0_256KB"); // 256KB SRAM scratchpad
  auto cache_memory = std::make_shared<scaleout::resources::MemoryCapacity>(
      64 * 1024, "SPAD_1_64KB"); // 64KB SRAM scratchpad

  // Add resources to the manager
  manager.addResource(read_port);
  manager.addResource(write_port);
  manager.addResource(rw_port);
  manager.addResource(main_memory);
  manager.addResource(cache_memory);

  // Display resource information
  std::cout << "\nCreated Resources:" << std::endl;
  std::cout << read_port->toString() << std::endl;
  std::cout << write_port->toString() << std::endl;
  std::cout << rw_port->toString() << std::endl;
  std::cout << main_memory->toString() << std::endl;
  std::cout << cache_memory->toString() << std::endl;

  // Show resource statistics
  auto stats = manager.getResourceStatistics();
  std::cout << "\nResource Statistics:" << std::endl;
  for (const auto &stat : stats) {
    std::cout << stat.first << ": " << stat.second << " instances" << std::endl;
  }

  // Demonstrate resource usage
  std::cout << "\n=== Resource Usage Demonstration ===" << std::endl;

  // Use SRAM ports
  if (read_port->isAvailable()) {
    std::cout << "Acquiring read port..." << std::endl;
    read_port->acquire();
    std::cout << "Read port available: " << read_port->isAvailable()
              << std::endl;
  }

  // Use SRAM capacity
  std::cout << "SPAD_0 utilization: " << main_memory->getUtilizationPercentage()
            << "%" << std::endl;

  if (main_memory->canConsume(32 * 1024)) { // 32KB
    main_memory->consume(32 * 1024);
    std::cout << "Consumed 32KB, new utilization: "
              << main_memory->getUtilizationPercentage() << "%" << std::endl;
  }

  // Find resources by ID
  auto found_resource = manager.findResource(read_port->getResourceId());
  if (found_resource) {
    std::cout << "Found resource by ID: " << found_resource->toString()
              << std::endl;
  }

  // Reset resources
  read_port->reset();
  main_memory->reset();
  std::cout << "\nAfter reset:" << std::endl;
  std::cout << "Read port available: " << read_port->isAvailable() << std::endl;
  std::cout << "Main memory utilization: "
            << main_memory->getUtilizationPercentage() << "%" << std::endl;
}

/**
 * @brief Demonstrate advanced resource management features
 */
void demonstrateAdvancedUsage() {
  std::cout << "\n=== Advanced Resource Management ===" << std::endl;

  auto &manager = scaleout::resources::ResourceManager::getInstance();

  // Create multiple resources of the same type
  std::vector<std::shared_ptr<scaleout::resources::MemoryPort>> new_ports;
  for (int i = 0; i < 3; ++i) {
    auto port = std::make_shared<scaleout::resources::MemoryPort>(
        scaleout::resources::MemoryPort::PortType::READ_WRITE, 32,
        "Port_" + std::to_string(i));
    manager.addResource(port);
    new_ports.push_back(port);
  }

  // Get all resources and filter for MemoryPort
  auto all_resources = manager.getAllResources();
  std::vector<std::shared_ptr<scaleout::resources::MemoryPort>> all_ports;
  for (const auto &resource : all_resources) {
    if (resource->getResourceTypeName() == "MemoryPort") {
      auto port =
          std::dynamic_pointer_cast<scaleout::resources::MemoryPort>(resource);
      if (port) {
        all_ports.push_back(port);
      }
    }
  }
  std::cout << "Total MemoryPort instances: " << all_ports.size() << std::endl;

  // Reserve all available ports
  int reserved_count = 0;
  for (auto &port : all_ports) {
    if (port->isAvailable() && port->acquire()) {
      reserved_count++;
    }
  }
  std::cout << "Reserved " << reserved_count << " ports" << std::endl;

  // Show availability status
  std::cout << "Port availability status:" << std::endl;
  for (const auto &port : all_ports) {
    std::cout << "  " << port->getResourceName() << ": "
              << (port->isAvailable() ? "Available" : "Reserved") << std::endl;
  }

  // Clean up - remove some resources
  if (!all_ports.empty()) {
    uint64_t id_to_remove = all_ports[0]->getResourceId();
    if (manager.removeResource(id_to_remove)) {
      std::cout << "Removed resource with ID: " << id_to_remove << std::endl;
    }
  }

  // Count remaining MemoryPort instances
  auto remaining_resources = manager.getAllResources();
  int memory_port_count = 0;
  for (const auto &resource : remaining_resources) {
    if (resource->getResourceTypeName() == "MemoryPort") {
      memory_port_count++;
    }
  }
  std::cout << "Remaining MemoryPort instances: " << memory_port_count
            << std::endl;
}

int main() {
  std::cout << "======================================" << std::endl;
  std::cout << "  LOOM Resource Management Demo" << std::endl;
  std::cout << "======================================" << std::endl;
  std::cout << std::endl;

  try {
    // Run the basic resource management demonstration
    demonstrateBasicUsage();

    // Run the advanced resource management demonstration
    demonstrateAdvancedUsage();

    std::cout << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "  Resource Demo Completed Successfully!" << std::endl;
    std::cout << "======================================" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << std::endl;
    std::cerr << "======================================" << std::endl;
    std::cerr << "  Error in Resource Management Demo" << std::endl;
    std::cerr << "======================================" << std::endl;
    std::cerr << "Error details: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << std::endl;
    std::cerr << "======================================" << std::endl;
    std::cerr << "  Unknown Error in Resource Demo" << std::endl;
    std::cerr << "======================================" << std::endl;
    return 1;
  }

  return 0;
}