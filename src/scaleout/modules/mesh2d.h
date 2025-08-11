#pragma once

#include "../resources/chain.h"
#include "module.h"
#include <cstddef>
#include <utility>
#include <vector>

namespace scaleout {
namespace modules {

/**
 * 2D Mesh module composed of horizontal and vertical chains.
 *
 * A Mesh2D of size (num_rows, num_cols) is represented by num_rows horizontal
 * chains (left-right) and num_cols vertical chains (top-down). Reduction is
 * not supported here by design; use Torus for that.
 */
class Mesh2D : public Module {
private:
  size_t num_rows_;
  size_t num_cols_;
  std::vector<resources::Chain *> horizontal_chains_;
  std::vector<resources::Chain *> vertical_chains_;

public:
  Mesh2D(size_t num_rows, size_t num_cols, std::vector<int> core_ids,
         std::vector<resources::Chain *> horizontal_chains,
         std::vector<resources::Chain *> vertical_chains,
         const std::string &name = "Mesh2D")
      : Module(name, std::move(core_ids)), num_rows_(num_rows),
        num_cols_(num_cols), horizontal_chains_(std::move(horizontal_chains)),
        vertical_chains_(std::move(vertical_chains)) {}

  std::string getTypeName() const override { return "Mesh2D"; }

  size_t getNumRows() const { return num_rows_; }
  size_t getNumCols() const { return num_cols_; }

  bool isAvailable() const {
    // Mesh is available if all underlying chains are free
    for (const auto *c : horizontal_chains_) {
      if (c == nullptr || !c->isAvailable()) {
        return false;
      }
    }
    for (const auto *c : vertical_chains_) {
      if (c == nullptr || !c->isAvailable()) {
        return false;
      }
    }
    return true;
  }

  bool acquire() {
    if (!isAvailable()) {
      return false;
    }
    for (auto *c : horizontal_chains_) {
      if (!c->consume()) {
        // Rollback already acquired chains
        for (auto *r : horizontal_chains_) {
          if (r == c)
            break;
          r->release();
        }
        return false;
      }
    }
    for (auto *c : vertical_chains_) {
      if (!c->consume()) {
        // Rollback all horizontal and prior vertical
        for (auto *r : horizontal_chains_) {
          r->release();
        }
        for (auto *r : vertical_chains_) {
          if (r == c)
            break;
          r->release();
        }
        return false;
      }
    }
    return true;
  }

  void release() {
    for (auto *c : horizontal_chains_) {
      c->release();
    }
    for (auto *c : vertical_chains_) {
      c->release();
    }
  }
};

} // namespace modules
} // namespace scaleout
