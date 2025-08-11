#include "scaleout/modules/mesh2d.h"
#include "scaleout/modules/torus.h"
#include "scaleout/resources/chain.h"
#include "scaleout/resources/memory.h"
#include "scaleout/resources/ring.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace scaleout;

static void printHeader(const std::string &title) {
  std::cout << "\n======================================" << std::endl;
  std::cout << "  " << title << std::endl;
  std::cout << "======================================" << std::endl;
}

static void demonstrateTorusWithMemoryBanks() {
  printHeader("Torus Module - Broadcast with MemoryBank");

  // Prepare ring and cores
  resources::Ring ring("MainRing");
  std::vector<int> core_ids{0, 1, 2, 3};

  // Create per-core memory banks (capacity + read and write ports)
  std::vector<std::unique_ptr<resources::MemoryCapacity>> capacities;
  std::vector<std::unique_ptr<resources::MemoryPort>> read_ports;
  std::vector<std::unique_ptr<resources::MemoryPort>> write_ports;
  std::vector<std::unique_ptr<resources::MemoryBank>> banks_storage;
  std::vector<resources::MemoryBank *> banks;

  for (size_t i = 0; i < core_ids.size(); ++i) {
    capacities.emplace_back(std::make_unique<resources::MemoryCapacity>(
        1024, "SPAD_" + std::to_string(i)));
    read_ports.emplace_back(std::make_unique<resources::MemoryPort>(
        resources::MemoryPort::PortType::READ, 64,
        "RPORT_" + std::to_string(i)));
    write_ports.emplace_back(std::make_unique<resources::MemoryPort>(
        resources::MemoryPort::PortType::WRITE, 64,
        "WPORT_" + std::to_string(i)));
    banks_storage.emplace_back(std::make_unique<resources::MemoryBank>(
        capacities.back().get(), read_ports.back().get(),
        write_ports.back().get()));
    banks.push_back(banks_storage.back().get());
  }

  modules::Torus torus(core_ids, ring, /*per_core_memory*/ {},
                       /*per_core_memory_banks*/ banks);

  const size_t element_bytes = 4; // Broadcasting 4 bytes

  std::cout << "Ring available before: " << ring.isAvailable() << std::endl;
  std::cout << "canBroadcast? " << torus.canBroadcast(element_bytes)
            << std::endl;
  bool ok = torus.acquireForBroadcast(element_bytes);
  std::cout << "acquireForBroadcast -> " << ok << std::endl;
  std::cout << "Ring available after acquire: " << ring.isAvailable()
            << std::endl;

  // Validate all banks consumed ports and capacity
  for (size_t i = 0; i < banks.size(); ++i) {
    auto *cap = banks[i]->getCapacity();
    auto *rp = banks[i]->getReadPort();
    auto *wp = banks[i]->getWritePort();
    std::cout << "Core " << i << " used=" << cap->getUsedSize()
              << "B, R-available=" << rp->isAvailable()
              << ", W-available=" << wp->isAvailable() << std::endl;
  }

  torus.releaseBroadcast(element_bytes);
  std::cout << "Ring available after release: " << ring.isAvailable()
            << std::endl;
  for (size_t i = 0; i < banks.size(); ++i) {
    auto *cap = banks[i]->getCapacity();
    auto *rp = banks[i]->getReadPort();
    auto *wp = banks[i]->getWritePort();
    std::cout << "Core " << i << " used=" << cap->getUsedSize()
              << "B, R-available=" << rp->isAvailable()
              << ", W-available=" << wp->isAvailable() << std::endl;
  }

  // Demonstrate reduction acquire/release
  std::cout << "\nReduction acquire: " << torus.acquireForReduction()
            << std::endl;
  std::cout << "Ring available during reduction: " << ring.isAvailable()
            << std::endl;
  torus.releaseReduction();
  std::cout << "Ring available after reduction: " << ring.isAvailable()
            << std::endl;
}

static void demonstrateTorusCapacityOnly() {
  printHeader("Torus Module - Broadcast with Capacity Only");

  resources::Ring ring("SecondaryRing");
  std::vector<int> core_ids{0, 1, 2};

  // Only capacities per core
  std::vector<std::unique_ptr<resources::MemoryCapacity>> capacities;
  std::vector<resources::MemoryCapacity *> capacity_ptrs;
  for (size_t i = 0; i < core_ids.size(); ++i) {
    capacities.emplace_back(std::make_unique<resources::MemoryCapacity>(
        128, "SPAD_Capacity_" + std::to_string(i)));
    capacity_ptrs.push_back(capacities.back().get());
  }

  modules::Torus torus(core_ids, ring, capacity_ptrs, /*banks*/ {});
  const size_t bytes = 8;

  std::cout << "Ring available before: " << ring.isAvailable() << std::endl;
  std::cout << "canBroadcast? " << torus.canBroadcast(bytes) << std::endl;
  bool ok = torus.acquireForBroadcast(bytes);
  std::cout << "acquireForBroadcast -> " << ok << std::endl;
  for (size_t i = 0; i < capacity_ptrs.size(); ++i) {
    std::cout << "Core " << i << " used=" << capacity_ptrs[i]->getUsedSize()
              << "B" << std::endl;
  }
  torus.releaseBroadcast(bytes);
}

static void demonstrateMesh2D() {
  printHeader("Mesh2D Module - Acquire/Release");

  // Build a 2x2 mesh
  size_t rows = 2, cols = 2;
  std::vector<int> core_ids{0, 1, 2, 3};
  resources::Chain h0("H0"), h1("H1");
  resources::Chain v0("V0"), v1("V1");
  std::vector<resources::Chain *> horizontal{&h0, &h1};
  std::vector<resources::Chain *> vertical{&v0, &v1};

  modules::Mesh2D mesh(rows, cols, core_ids, horizontal, vertical);

  std::cout << "Mesh available before: " << mesh.isAvailable() << std::endl;
  bool ok = mesh.acquire();
  std::cout << "mesh.acquire() -> " << ok << std::endl;
  std::cout << "H0 avail=" << h0.isAvailable()
            << ", H1 avail=" << h1.isAvailable()
            << ", V0 avail=" << v0.isAvailable()
            << ", V1 avail=" << v1.isAvailable() << std::endl;
  mesh.release();
  std::cout << "Mesh available after release: " << mesh.isAvailable()
            << std::endl;
}

int main() {
  try {
    demonstrateTorusWithMemoryBanks();
    demonstrateTorusCapacityOnly();
    demonstrateMesh2D();
  } catch (const std::exception &e) {
    std::cerr << "Error in module demo: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown error in module demo" << std::endl;
    return 1;
  }
  return 0;
}
