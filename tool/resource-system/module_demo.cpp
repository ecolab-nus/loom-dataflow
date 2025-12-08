#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "chain.h"  // modules::Chain (includes resources::Chain)
#include "mesh2d.h"
#include "torus.h"
#include "memory.h"
#include "ring.h"
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

  mlir::MLIRContext ctx;
  auto dim = mlir::getAffineDimExpr(0, &ctx);
  mlir::AffineMap torusMap = mlir::AffineMap::get(1, 0, dim);

  modules::Torus torus("TorusBanks", core_ids, torusMap, ring,
                       /*per_core_memory*/ {},
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

  mlir::MLIRContext ctx;
  auto dim = mlir::getAffineDimExpr(0, &ctx);
  mlir::AffineMap torusMap = mlir::AffineMap::get(1, 0, dim);

  modules::Torus torus("TorusCapacity", core_ids, torusMap, ring, capacity_ptrs,
                       /*banks*/ {});
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
  // Build resources
  resources::Chain h0_r("H0"), h1_r("H1");
  resources::Chain v0_r("V0"), v1_r("V1");

  // Build module chains per row/col with identity 1D maps
  mlir::MLIRContext ctx;
  auto d = mlir::getAffineDimExpr(0, &ctx);
  mlir::AffineMap oneD = mlir::AffineMap::get(1, 0, d);

  std::vector<int> row0{0, 1};
  std::vector<int> row1{2, 3};
  std::vector<int> col0{0, 2};
  std::vector<int> col1{1, 3};

  modules::Chain h0("H0mod", row0, oneD, h0_r);
  modules::Chain h1("H1mod", row1, oneD, h1_r);
  modules::Chain v0("V0mod", col0, oneD, v0_r);
  modules::Chain v1("V1mod", col1, oneD, v1_r);

  std::vector<modules::Chain *> horizontal{&h0, &h1};
  std::vector<modules::Chain *> vertical{&v0, &v1};

  // Mesh mapping: (i,j)[N] -> i*N + j
  auto i = mlir::getAffineDimExpr(0, &ctx);
  auto j = mlir::getAffineDimExpr(1, &ctx);
  auto N = mlir::getAffineSymbolExpr(0, &ctx);
  mlir::AffineMap meshMap = mlir::AffineMap::get(2, 1, i * N + j);

  modules::Mesh2D mesh("Mesh2x2", core_ids, meshMap, rows, cols, horizontal,
                       vertical);

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
