#pragma once

#include "hardware_info.h"
#include "llvm/ADT/SmallVector.h"

namespace loom {

/**
 * \brief Enumerate mapping candidates for spatial dimensions.
 */
class MappingEnumerator {
public:
  MappingEnumerator(const HardwareInfo &hardwareInfo)
      : hardwareInfo(hardwareInfo) {}

  /**
   * \brief Generate all possible bucketing combinations of parallel iterators
   * to hardware dimensions.
   * \param numParallelIter Number of parallel iterators in the root loop.
   * \param numHWDims Number of available hardware spatial dimensions.
   */
  llvm::SmallVector<DimBuckets>
  generateAllPossibleBuckets(unsigned numParallelIter, unsigned numHWDims);

  /**
   * \brief Permute the buckets based on hardware info constraints.
   */
  llvm::SmallVector<DimBuckets> permuteBuckets(const DimBuckets &baseBuckets);

private:
  const HardwareInfo &hardwareInfo;

  void enumerateBucketingRec(unsigned dimIdx, unsigned numHWDims,
                             DimBuckets &currentBuckets,
                             llvm::SmallVector<DimBuckets> &out);

  void
  cartesianProductOfBuckets(unsigned iterIdx, DimBuckets &current,
                            const llvm::SmallVector<DimBuckets> &bucketsPerIter,
                            llvm::SmallVector<DimBuckets> &out);

  llvm::SmallVector<DimBuckets> generateAllPossibleMappings(
      const llvm::SmallVector<DimBuckets> &permutedBucketsPerIter);
};

} // namespace loom
