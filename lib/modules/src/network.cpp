#include "../inc/network.h"
#include "mlir/IR/AffineMap.h"

namespace scaleout {
namespace modules {

struct NetworkModule::Impl {
  mlir::AffineMap coreIndexMap;
};

NetworkModule::NetworkModule(std::string name, std::vector<int> core_ids,
                             const mlir::AffineMap &coreIndexMap)
    : Module(std::move(name)), core_ids_(std::move(core_ids)),
      impl_(std::make_unique<Impl>()) {
  impl_->coreIndexMap = coreIndexMap;
}

NetworkModule::~NetworkModule() = default;

NetworkModule::NetworkModule(NetworkModule &&other) noexcept = default;
NetworkModule &
NetworkModule::operator=(NetworkModule &&other) noexcept = default;

const mlir::AffineMap &NetworkModule::getAffinePlacementMap() const {
  return impl_->coreIndexMap;
}

} // namespace modules
} // namespace scaleout
