#include "mapping_enumeration.h"
#include <algorithm>

namespace loom {

void MappingEnumerator::enumerateBucketingRec(
    unsigned dimIdx, unsigned numHWDims, DimBuckets &currentBuckets,
    llvm::SmallVector<DimBuckets> &out) {
  if (dimIdx == numHWDims) {
    out.push_back(currentBuckets);
    return;
  }
  for (unsigned it = 0; it < currentBuckets.size(); ++it) {
    currentBuckets[it].push_back(dimIdx);
    enumerateBucketingRec(dimIdx + 1, numHWDims, currentBuckets, out);
    currentBuckets[it].pop_back();
  }
}

llvm::SmallVector<DimBuckets>
MappingEnumerator::generateAllPossibleBuckets(unsigned numParallelIter,
                                              unsigned numHWDims) {
  llvm::SmallVector<DimBuckets> bucketing_results;
  DimBuckets currentBuckets(numParallelIter);
  enumerateBucketingRec(0, numHWDims, currentBuckets, bucketing_results);
  return bucketing_results;
}

void MappingEnumerator::cartesianProductOfBuckets(
    unsigned iterIdx, DimBuckets &current,
    const llvm::SmallVector<DimBuckets> &bucketsPerIter,
    llvm::SmallVector<DimBuckets> &out) {
  if (iterIdx == current.size()) {
    out.push_back(current);
    return;
  }

  const auto &buckets = bucketsPerIter[iterIdx];
  auto saved = current[iterIdx];

  if (buckets.empty()) {
    current[iterIdx].clear();
    cartesianProductOfBuckets(iterIdx + 1, current, bucketsPerIter, out);
    current[iterIdx] = saved;
    return;
  }

  for (const auto &dims : buckets) {
    current[iterIdx] = dims;
    cartesianProductOfBuckets(iterIdx + 1, current, bucketsPerIter, out);
  }

  current[iterIdx] = saved;
}

llvm::SmallVector<DimBuckets> MappingEnumerator::generateAllPossibleMappings(
    const llvm::SmallVector<DimBuckets> &permutedBucketsPerIter) {
  llvm::SmallVector<DimBuckets> result;
  DimBuckets currentBuckets(permutedBucketsPerIter.size());
  cartesianProductOfBuckets(0, currentBuckets, permutedBucketsPerIter, result);
  return result;
}

llvm::SmallVector<DimBuckets>
MappingEnumerator::permuteBuckets(const DimBuckets &baseBuckets) {
  const unsigned numIters = static_cast<unsigned>(baseBuckets.size());
  llvm::SmallVector<DimBuckets> permutedBucketsPerIter(numIters);

  for (unsigned it = 0; it < numIters; ++it) {
    llvm::SmallVector<unsigned> dims = baseBuckets[it];
    if (dims.size() <= 1) {
      permutedBucketsPerIter[it].push_back(dims);
    } else {
      std::sort(dims.begin(), dims.end());
      do {
        permutedBucketsPerIter[it].push_back(dims);
      } while (std::next_permutation(dims.begin(), dims.end()));
    }
  }

  return generateAllPossibleMappings(permutedBucketsPerIter);
}

} // namespace loom
