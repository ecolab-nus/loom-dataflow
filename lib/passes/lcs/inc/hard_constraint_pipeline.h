#pragma once

namespace loom {
namespace lcs {

class HWOpRegistry;
struct ConstraintScope;

class HardConstraintPipeline {
public:
  static void pushAll(const HWOpRegistry *registry, ConstraintScope &scope);
};

} // namespace lcs
} // namespace loom
