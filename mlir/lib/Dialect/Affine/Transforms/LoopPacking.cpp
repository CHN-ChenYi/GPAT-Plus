//===- LoopPacking.cpp --- Loop packing pass ----------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to pack loop nests when profitable for CPUs
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using llvm::dbgs;
using llvm::SmallMapVector;
using llvm::SmallSet;

#define DEBUG_TYPE "affine-loop-pack"

namespace {

/// A pass to perform loop packing on suitable packing candidates of a loop
/// nest.
struct LoopPacking : public AffineLoopPackingBase<LoopPacking> {
  LoopPacking() = default;

  void runOnOperation() override;
  void runOnOuterForOp(AffineForOp outerForOp,
                       DenseSet<Operation *> &copyNests);
};

} // end anonymous namespace

/// Creates a pass to perform loop packing
std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLoopPackingPass() {
  return std::make_unique<LoopPacking>();
}

/// Returns the mininum trip count of the loop if at least one map
/// element is a constant, None otherwise.
/// Loops should be normalized before using this function
static Optional<uint64_t> getApproxTripCount(AffineForOp forOp) {
  SmallVector<Value> operands;
  AffineMap map;
  getTripCountMapAndOperands(forOp, &map, &operands);

  if (!map)
    return None;

  // Take the min contant trip count
  Optional<uint64_t> tripCount;
  for (auto resultExpr : map.getResults()) {
    if (auto constExpr = resultExpr.dyn_cast<AffineConstantExpr>()) {
      if (tripCount.hasValue())
        tripCount = std::min(tripCount.getValue(),
                             static_cast<uint64_t>(constExpr.getValue()));
      else
        tripCount = constExpr.getValue();
    }
  }
  return tripCount;
}

/// Checks if a memRefRegion is contiguous within a tile.
/// Given a tile shape, after dropping consecutive outermost dimensions of size
/// '1', and consecutive innermost dimentions of full size (equal to the memRef
/// shape), if the resulting shape is >= 2D, i.e. matrix or above, the memref
/// accesses are not all contiguous within the tile
static bool isTileShapeContiguous(ArrayRef<int64_t> memRefShape,
                                  ArrayRef<int64_t> tileShape) {
  assert(memRefShape.size() == tileShape.size());
  int begin = 0;
  int end = memRefShape.size() - 1;
  int innerDims = 0;

  // 'Drop' the 1's from the outermost dimensions until a non-1 is encountered.
  while (begin <= end && tileShape[begin] == 1)
    begin++;

  // If the entire tile is ones then it is a single element access in the
  // structure and thus contiguous.
  if (begin == (int)memRefShape.size())
    return true;

  // 'Drop' the full size dimensions from the innermost dimensions until a
  // non-full is encountered.
  // TODO: maybe this check could be done even if the values are symbolic
  while (0 <= end && tileShape[end] == memRefShape[end])
    end--;

  // If the tile shape completely matches the memref region shape then the
  // accesses are continuous.
  if (end == -1)
    return true;

  // After 'dropping' according to the conditions above, if the resulting shape
  // is not a vector (1D) then the memref accesses are not contiguous.
  innerDims = end - begin + 1;
  return innerDims < 2;
}

static uint64_t approxCacheEntriesNeeded(MemRefType memRefType,
                                       ArrayRef<int64_t> memRefShape,
                                       ArrayRef<int64_t> tileShape,
                                       uint64_t entrySizeInB) {
  assert(memRefShape.size() == tileShape.size());
  int typeSizeBytes = getMemRefEltSizeInBytes(memRefType);
  // Number of elements addresed in one TLB entry
  int elementsInEntry = floorDiv(entrySizeInB, typeSizeBytes);
  int entriesNeeded = 1;

  // Find contiguous portion of the tile, the innemost dimension is always
  // included
  int contiguousDim = memRefShape.size() - 1;
  while (0 <= contiguousDim &&
         tileShape[contiguousDim] == memRefShape[contiguousDim])
    contiguousDim--;
  // Make contiguousDim point to the outermost contiguous dimension
  if (contiguousDim != (int)memRefShape.size() - 1)
    contiguousDim += 1;

  // Count elements in contiguous portion
  int contiguousElems = 1;
  for (size_t idx = contiguousDim; idx < memRefShape.size(); ++idx)
    contiguousElems *= tileShape[idx];
  // Get number of pages needed to address contiguous portion of the tile
  entriesNeeded *= ceilDiv(contiguousElems, elementsInEntry);

  int remainingDims = contiguousDim - 1;
  while (remainingDims >= 0) {
    int entriesNeededMemRef = 1;
    int offset = 1;
    // Offset between elements in contiguous portion of tile
    for (size_t i = remainingDims + 1; i < memRefShape.size(); i++)
      offset *= memRefShape[i];
    // Check if a single page can address multiple contiguous portions of a tile
    // at once e.g. if there are 512 element per entry, memref has dimensions
    // 256x256, and tile 6x5, 3 entries are needed as each entry can address 2
    // full rows of the memref
    entriesNeededMemRef =
        ceilDiv(tileShape[remainingDims] * offset, elementsInEntry);
    // Otherwise compute the number of entries needed for the contiguous portion
    // (already added to entriesNeeded) and multiply by the size of the tile
    // dimension
    entriesNeeded *= tileShape[remainingDims];
    remainingDims--;
    // If entriesNeededMemRef is smaller, consider the current remainingDim as
    // part of contiguous portion
    if (entriesNeededMemRef < entriesNeeded) {
      entriesNeeded = entriesNeededMemRef;
    } else {
      break;
    }
  }

  // Multiply entries needed by the size of the remaining outer tile dimensions
  while (remainingDims >= 0) {
    entriesNeeded *= tileShape[remainingDims];
    remainingDims--;
  }

  return entriesNeeded;
}

/// Given a forOp, this function will collect the memrefs of loads or stores.
/// Only collects memRefs with rank greater or equal to minRank.
static void getMemRefsInForOp(AffineForOp forOp, SetVector<Value> &memRefs,
                              unsigned minRank = 2) {
  memRefs.clear();
  forOp.walk([&](Operation *op) {
    Value memRef;
    if (auto load = dyn_cast<AffineLoadOp>(op))
      memRef = load.getMemRef();
    else if (auto store = dyn_cast<AffineStoreOp>(op))
      memRef = store.getMemRef();
    else
      return WalkResult::advance();

    unsigned rank = memRef.getType().cast<MemRefType>().getRank();
    if (rank >= minRank)
      memRefs.insert(memRef);

    return WalkResult::advance();
  });
}

/// True if a loop is a parent of an operation.
static bool isLoopParentOfOp(AffineForOp forOp, Operation *op) {
  Operation *currOp = op;

  while ((currOp = currOp->getParentOp())) {
    if (auto currFor = dyn_cast<AffineForOp>(currOp)) {
      if (currFor == forOp)
        return true;
    }
  }
  return false;
}

/// Check if a memRef is used in a forOp
static bool isMemRefUsedInForOp(AffineForOp forOp, Value memRef) {
  auto walkRes = forOp.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      if (memRef == loadOp.getMemRef())
        return WalkResult::interrupt();
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      if (memRef == storeOp.getMemRef())
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return walkRes.wasInterrupted();
}

/// Check if an AffineForOp is invariant to all loads and stores of memRef
static bool isLoopInvariantToMemRef(AffineForOp forOp, Value memRef) {

  SmallVector<AffineLoadOp> loads;
  SmallVector<AffineStoreOp> stores;

  // Gather loads and stores
  forOp.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      if (memRef == loadOp.getMemRef())
        loads.push_back(loadOp);
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      if (memRef == storeOp.getMemRef())
        stores.push_back(storeOp);
    }
  });

  // If there are no loads and stores
  if (loads.empty() && stores.empty())
    return true;

  // Collects indices used in all loads and stores
  DenseSet<Value> accessIndices;
  for (AffineLoadOp load : loads)
    for (Value operand : load.indices())
      accessIndices.insert(operand);
  for (AffineStoreOp store : stores)
    for (Value operand : store.indices())
      accessIndices.insert(operand);

  // Collect thisLoopIV of forOp and add dependent loop IVs:
  // If thisLoopIV is used in any other inner loops as an upper or lower bound
  // operand, add the loop IV of these inner loops as a dependent loop IVs. Then
  // add dependent loop IVs of the added loop IVs.
  DenseSet<Value> loopIVs;
  DenseSet<Value> checkDepsList;
  auto thisLoopIV = forOp.getInductionVar();
  checkDepsList.insert(thisLoopIV);

  // Check dependencies and then move loop IV to loopIVs
  while (!checkDepsList.empty()) {
    Value currLoopIV = *checkDepsList.begin();
    AffineForOp currForOp = getForInductionVarOwner(currLoopIV);

    currForOp.walk([&](AffineForOp walkForOp) {
      if (walkForOp == currForOp)
        return WalkResult::advance();

      for (auto upOperand : walkForOp.getUpperBoundOperands()) {
        if (currLoopIV == upOperand) {
          checkDepsList.insert(walkForOp.getInductionVar());
          return WalkResult::advance();
        }
      }
      for (auto lbOperand : walkForOp.getLowerBoundOperands()) {
        if (currLoopIV == lbOperand) {
          checkDepsList.insert(walkForOp.getInductionVar());
          return WalkResult::advance();
        }
      }
      return WalkResult::advance();
    });

    checkDepsList.erase(currLoopIV);
    loopIVs.insert(currLoopIV);
  }

  // Get accesses that are invariant to all loop operands
  DenseSet<Value> invariantAccesses;
  for (auto accessVal : accessIndices) {
    bool isInvariant = true;

    for (auto loopVal : loopIVs) {
      if (!isAccessIndexInvariant(loopVal, accessVal)) {
        isInvariant = false;
        break;
      }
    }

    if (isInvariant)
      invariantAccesses.insert(accessVal);
  }

  // If all acesses are invariant, the loop is invariant to the memref
  return invariantAccesses.size() == accessIndices.size();
}

/// Computes memRefRegions for a memRef within a forOp.
static void createMemRefRegions(AffineForOp forOp, Value memRef,
                                std::unique_ptr<MemRefRegion> &readRegion,
                                std::unique_ptr<MemRefRegion> &writeRegion) {
  // To check for errors when walking the block.
  bool error = false;
  unsigned depth = getNestingDepth(forOp);

  // Walk this range of operations to gather all memory regions.
  forOp.walk([&](Operation *op) {
    // Skip ops that are not loads and stores of memRef
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      if (memRef != loadOp.getMemRef())
        return WalkResult::advance();
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      if (memRef != storeOp.getMemRef())
        return WalkResult::advance();
    } else {
      return WalkResult::advance();
    }

    // Compute the MemRefRegion accessed.
    auto region = std::make_unique<MemRefRegion>(op->getLoc());
    if (failed(region->compute(op, depth, /*sliceState=*/nullptr,
                               /*addMemRefDimBounds=*/false))) {
      LLVM_DEBUG(dbgs() << "[DEBUG] Error obtaining memory region of " << memRef
                        << " : semi-affine maps?\n");
      error = true;
      return WalkResult::interrupt();
    }

    // Attempts to update regions
    auto updateRegion = [&](const std::unique_ptr<MemRefRegion> &targetRegion) {
      if (!targetRegion) {
        return;
      }

      // Perform a union with the existing region.
      if (failed(targetRegion->unionBoundingBox(*region))) {
        LLVM_DEBUG(dbgs() << "[DEBUG] Error obtaining memory region of "
                          << memRef << " : semi-affine maps?\n");
        error = true;
        return;
      }
      // Union was computed and stored in 'targetRegion': copy to 'region'.
      region->getConstraints()->clearAndCopyFrom(
          *targetRegion->getConstraints());
    };

    // Update region if region already exists
    updateRegion(readRegion);
    if (error)
      return WalkResult::interrupt();
    updateRegion(writeRegion);
    if (error)
      return WalkResult::interrupt();

    // Add region if region is empty
    if (region->isWrite() && !writeRegion) {
      writeRegion = std::move(region);
    } else if (!region->isWrite() && !readRegion) {
      readRegion = std::move(region);
    }

    return WalkResult::advance();
  });

  if (error) {
    Block::iterator begin = Block::iterator(forOp);
    begin->emitWarning("Creating and updating regions failed in this block\n");
    // clean regions
    readRegion = nullptr;
    writeRegion = nullptr;
  }
}

/// Get tile shape of memRef given a memRefRegion
/// result is returned in tileShape
static void computeTileShape(Value memRef,
                             std::unique_ptr<MemRefRegion> &region,
                             llvm::SmallVectorImpl<int64_t> &tileShape) {
  auto memRefType = memRef.getType().cast<MemRefType>();
  unsigned rank = memRefType.getRank();
  tileShape.clear();

  // Compute the extents of the buffer.
  std::vector<SmallVector<int64_t, 4>> lbs;
  SmallVector<int64_t, 8> lbDivisors;
  lbs.reserve(rank);
  Optional<int64_t> numElements =
      region->getConstantBoundingSizeAndShape(&tileShape, &lbs, &lbDivisors);
  if (!numElements.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "Non-constant region size not supported\n");
  }
}

// Tries to find operand in expression only recursing in add or sub expressions
// operandIdx is the index of the operand to be found
// expr is the result expression of an access map
// numDims is the number of dimensions in the access map
//
// static
// bool findOperandNonStrided(unsigned operandIdx, AffineExpr expr, unsigned
// numDims) {
//   // check if found operand
//   if (auto dimExpr = expr.dyn_cast<AffineDimExpr>())
//     return dimExpr.getPosition() == operandIdx;
//   if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>())
//     return numDims + symExpr.getPosition() == operandIdx;

//   // can only find operant recursing in add or sub expressions
//   if (auto binExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
//     if (binExpr.getKind() == AffineExprKind::Add) {
//       return findOperandNonStrided(operandIdx, binExpr.getLHS(), numDims) ||
//              findOperandNonStrided(operandIdx, binExpr.getRHS(), numDims);
//     }
//     // subtraction is implemented as mul -1
//     if (binExpr.getKind() == AffineExprKind::Mul) {
//       if (binExpr.getRHS().getKind() == AffineExprKind::Constant &&
//           binExpr.getRHS().cast<AffineConstantExpr>().getValue() == -1)
//         return findOperandNonStrided(operandIdx, binExpr.getLHS(), numDims);

//       if (binExpr.getLHS().getKind() == AffineExprKind::Constant &&
//           binExpr.getLHS().cast<AffineConstantExpr>().getValue() == -1)
//         return findOperandNonStrided(operandIdx, binExpr.getRHS(), numDims);
//     }
//   }
//   return false;
// }

// Counts how many times a operand is used in an expression
// operandIdx is the index of the operand to be found
// expr is the result expression of an access map
// numDims is the number of dimensions in the access map
//
// static
// unsigned countOperand(unsigned operandIdx, AffineExpr expr,
//                       unsigned numDims) {
//   // check if found operand
//   if (auto dimExpr = expr.dyn_cast<AffineDimExpr>())
//     if (dimExpr.getPosition() == operandIdx)
//       return 1;
//   if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>())
//     if (numDims + symExpr.getPosition() == operandIdx)
//       return 1;

//   if (auto binExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
//     return countOperand(operandIdx, binExpr.getLHS(), numDims) +
//            countOperand(operandIdx, binExpr.getRHS(), numDims);
//   }

//   return 0;
// }

// Checks if access is strided
// operandIdx is the index of the operand to be found
// resultExpr is the result expression of an access map
// numDims is the number of dimensions in the access map
//
// static
// bool isStridedAccess(unsigned operandIdx, AffineExpr resultExpr, unsigned
// numDims) {
//   unsigned count = countOperand(operandIdx, resultExpr, numDims);
//   // If the operand appears multiple times in the expression, it is strided
//   if (count > 1)
//     return true;
//   // If the operand appears once and could not be found only through add or
//   sub operations if (count == 1 && !findOperandNonStrided(operandIdx,
//   resultExpr, numDims))
//     return true;

//   return false;
// }

static LogicalResult generatePackings(
    Value memRef, Operation *forOp, std::unique_ptr<MemRefRegion> &readRegion,
    std::unique_ptr<MemRefRegion> &writeRegion,
    DenseSet<Operation *> &copyNests, AffineCopyOptions &copyOptions,
    ArrayRef<size_t> permutationOrder) {
  // Map from original memref's to the fast buffers that their accesses are
  // replaced with.
  DenseMap<Value, Value> fastBufferMap;

  Block::iterator begin = Block::iterator(forOp);
  Block::iterator end = std::next(Block::iterator(forOp));
  Block *block = begin->getBlock();

  LogicalResult ret = success();

  auto processRegions = [&](const std::unique_ptr<MemRefRegion> &region) {
    // pointer is null
    if (!region)
      return;

    // For each region, hoist copy in/out past all hoistable
    // 'affine.for's.
    Block::iterator copyInPlacementStart, copyOutPlacementStart;
    Block *copyPlacementBlock;

    copyInPlacementStart = begin;
    copyOutPlacementStart = end;
    copyPlacementBlock = block;

    uint64_t sizeInBytes;
    Block::iterator nBegin, nEnd;
    LogicalResult iRet = generateCopy(
        *region, block, begin, end, copyPlacementBlock, copyInPlacementStart,
        copyOutPlacementStart, copyOptions, fastBufferMap, copyNests,
        &sizeInBytes, &nBegin, &nEnd, permutationOrder);
    if (succeeded(iRet)) {
      // begin/end could have been invalidated, and need update.
      begin = nBegin;
      end = nEnd;
    } else {
      ret = failure();
    }
  };

  processRegions(readRegion);
  processRegions(writeRegion);

  return ret;
}

/// Stores information of an AffineForOp
class LoopInformation {
public:
  AffineForOp forOp;
  /// Depth of forOp
  uint depth;
  /// Constant trip count of forOp
  Optional<uint64_t> tripCount;
  /// Map from memref to its footprint in this forOp
  DenseMap<Value, Optional<int64_t>> memRefFootprintMap;
  /// Map from memref to a bool that says if the region is contiguous or not
  DenseMap<Value, bool> memRefIsContiguousMap;
  /// Map from memref to a it's tile shape in this loop
  DenseMap<Value, SmallVector<int64_t>> memRefTileShapeMap;
  /// Map from memref to it's read region
  DenseMap<Value, std::unique_ptr<MemRefRegion>> memRefReadRegion;
  /// Map from memref to it's write region
  DenseMap<Value, std::unique_ptr<MemRefRegion>> memRefWriteRegion;

  LoopInformation() = default;

  LoopInformation(AffineForOp forOp) : forOp{forOp} {
    this->setDepth();
    this->setTripCount();
  };

  Optional<int64_t> getMemRefFootprint(Value memRef) {
    if (this->memRefFootprintMap.count(memRef) == 0)
      memRefFootprintMap[memRef] =
          getMemoryFootprintBytesWithBranches(this->forOp, /*memorySpace*/ 0, memRef);

    return this->memRefFootprintMap[memRef];
  }

  // Only using read region as tile shape should be the same for write region
  ArrayRef<int64_t> getMemRefTileShape(Value memRef) {
    if (this->memRefTileShapeMap.count(memRef) == 0) {
      auto &region = this->getMemRefReadRegion(memRef);
      if (region) {
        computeTileShape(memRef, region, this->memRefTileShapeMap[memRef]);
      } else {
        this->memRefTileShapeMap[memRef] = SmallVector<int64_t>{};
      }
    }

    return this->memRefTileShapeMap[memRef];
  }

  bool getMemRefIsContiguous(Value memRef) {
    if (this->memRefIsContiguousMap.count(memRef) == 0) {
      auto memRefType = memRef.getType().cast<MemRefType>();
      auto memRefShape = memRefType.getShape();
      auto tileShape = this->getMemRefTileShape(memRef);
      if (tileShape.empty())
        return true;
      this->memRefIsContiguousMap[memRef] =
          isTileShapeContiguous(memRefShape, tileShape);
    }
    return this->memRefIsContiguousMap[memRef];
  }

  std::unique_ptr<MemRefRegion> &getMemRefReadRegion(Value memRef) {
    if (this->memRefReadRegion.count(memRef) == 0)
      createMemRefRegions(this->forOp, memRef, this->memRefReadRegion[memRef],
                          this->memRefWriteRegion[memRef]);

    return this->memRefReadRegion[memRef];
  }

  std::unique_ptr<MemRefRegion> &getMemRefWriteRegion(Value memRef) {
    if (this->memRefWriteRegion.count(memRef) == 0)
      createMemRefRegions(this->forOp, memRef, this->memRefReadRegion[memRef],
                          this->memRefWriteRegion[memRef]);

    return this->memRefWriteRegion[memRef];
  }

private:
  void setDepth() { this->depth = getNestingDepth(this->forOp); }

  void setTripCount() { this->tripCount = getApproxTripCount(this->forOp); }
};

/// Information of a possible packing
class PackingAttributes {
public:
  /// Packing candidate ID, for printing purposes
  int id;
  /// Target memRef
  Value memRef;
  /// LoopInfo of target forOp
  LoopInformation *loop;
  /// How many TLB entries are not needed by packing this loop
  uint64_t TLBImprovement;
  /// Footprint required in cache so that the packed buffer is not evited while
  /// reused
  Optional<uint64_t> residencyFootprint;
  /// Ratio that tries to approximate the benefit of packing this candidate
  Optional<double> gainCostRatio;
  /// Permutation vector, if not None, the packed buffer can be rearranged
  /// so its data layout correlates better with the current loop order
  Optional<SmallVector<size_t>> permutationOrder;
  /// True if permutation changes the indexing of an innermost loop IV
  bool innermostLoopIVPermutation;

  PackingAttributes(Value memRef, LoopInformation &loop)
      : id{-1}, memRef{memRef}, loop{&loop}, TLBImprovement{0},
        innermostLoopIVPermutation{false} {}

  // Ordering from based on gain/cost of packing ratio
  bool operator<(const PackingAttributes &other) const {
    if (this->gainCostRatio.hasValue() && other.gainCostRatio.hasValue()) {
      if (this->gainCostRatio.getValue() != other.gainCostRatio.getValue()) {
        return this->gainCostRatio.getValue() < other.gainCostRatio.getValue();
      }
    } else if (this->gainCostRatio.hasValue() && !other.gainCostRatio.hasValue()) {
      return false;
    } else if (!this->gainCostRatio.hasValue() && other.gainCostRatio.hasValue()) {
      return true;
    }
    return this->loop->depth > other.loop->depth;
  }

  bool operator>(const PackingAttributes &other) const { return other < *this; }

  void debug() {
    LLVM_DEBUG(dbgs() << "\n[Packing Attributes]\n");
    if (this->id != -1)
      LLVM_DEBUG(dbgs() << "    Id: " << this->id << "\n");
    LLVM_DEBUG(dbgs() << "    LoopIV location: "
                      << this->loop->forOp.getInductionVar().getLoc() << "\n");
    LLVM_DEBUG(dbgs() << "    MemRef: " << this->memRef << "\n");
    LLVM_DEBUG(dbgs() << "    Tile shape: ");
    for (auto i : this->loop->getMemRefTileShape(this->memRef)) {
      LLVM_DEBUG(dbgs() << i << " ");
    }
    LLVM_DEBUG(dbgs() << "\n");
    LLVM_DEBUG(dbgs() << "    Upper map: "
                      << this->loop->forOp.getUpperBoundMap() << "\n");
    LLVM_DEBUG(dbgs() << "    Lower map: "
                      << this->loop->forOp.getLowerBoundMap() << "\n");
    LLVM_DEBUG(dbgs() << "    Depth: " << this->loop->depth << "\n");
    LLVM_DEBUG(dbgs() << "    Contiguous: " << this->isContiguous() << "\n");
    LLVM_DEBUG(dbgs() << "    TripCount: "
                      << this->loop->tripCount.getValueOr(0) << "\n");
    LLVM_DEBUG(dbgs() << "    Footprint: " << this->getFootprint().getValueOr(0)
                      << "\n");
    LLVM_DEBUG(dbgs() << "    Residency footprint: "
                      << this->residencyFootprint.getValueOr(0) << "\n");
    LLVM_DEBUG(dbgs() << "    TLB improvement: " << this->TLBImprovement
                      << "\n");
    LLVM_DEBUG(dbgs() << "    GainCostRatio: "
                      << this->gainCostRatio.getValueOr(0) << "\n");
    if (this->permutationOrder.hasValue()) {
      LLVM_DEBUG(dbgs() << "    Permutation order: ");
      for (auto i : this->permutationOrder.getValue()) {
        LLVM_DEBUG(dbgs() << i << ",");
      }
      LLVM_DEBUG(dbgs() << "\n");
      LLVM_DEBUG(dbgs() << "    Permutes innermost: "
                        << this->innermostLoopIVPermutation << "\n");
    }
    LLVM_DEBUG(dbgs() << "\n");
  }

  /// Residency footprint is an approximation of space in cache necessary so the
  /// packing remains in cache The packing is reused at every iteration of the
  /// invariant forOp. It will remain in cache if there is enough space for the
  /// packing itself and for the other memrefs. Therefore residency footprint is
  /// calculated with:
  ///   - footprint of the packing
  ///   - 2 * footprint of other memrefs used in one iteration of the invariant
  ///   loop
  /// Twice the footprint of other memrefs so that the packing is not evicted in
  /// an LRU policy. If there are more children loops or if operations, this is
  /// an over approximation
  void setResidencyFootprint() {
    if (!this->getFootprint().hasValue()) {
      this->residencyFootprint = None;
      return;
    }

    uint64_t otherMemrefFootprint = 0;
    auto *forBodyBlock = this->loop->forOp.getBody();

    SetVector<Value> memRefs;
    getMemRefsInForOp(this->loop->forOp, memRefs, /*minRank=*/1);

    for (auto memRef : memRefs) {
      if (this->memRef == memRef)
        continue;

      auto footprint = getMemoryFootprintBytesWithBranches(this->loop->forOp,
          *forBodyBlock, forBodyBlock->begin(), forBodyBlock->end(), 0, memRef);

      LLVM_DEBUG(dbgs() << "[DEBUG] Footprint of " << memRef << ": "
                        << footprint.getValueOr(0) << "\n");
                    
      if (footprint.hasValue())
        otherMemrefFootprint += 2 * footprint.getValue();
      else
        return;
    }

    this->residencyFootprint =
        this->getFootprint().getValue() + otherMemrefFootprint;
  }

  /// Calculates how good this packing option is
  void setGainCostRatio(uint64_t cacheLineSizeInB) {
    // calculate gain: reuse factor (trip count) * TLB entries saved by this
    // packing
    double gain = this->TLBImprovement;

    // calculate packing cost: (read only) 1 * footprint or (read and write) 2 *
    // footprint
    double cost = 0.0;
    if (this->getMemRefReadRegion()) {
      cost++;
    }
    if (this->getMemRefWriteRegion()) {
      cost++;
    }

    // Get memRef shape
    auto memRefType = this->memRef.getType().cast<MemRefType>();
    auto tileShape = this->loop->getMemRefTileShape(this->memRef);
    if (tileShape.empty())
      return;
    ArrayRef<int64_t> memRefShape = memRefType.getShape();
    int typeSizeBytes = getMemRefEltSizeInBytes(memRefType);
    // Number of elements addresed in one cache line
    int elementsInCacheLine = floorDiv(cacheLineSizeInB, typeSizeBytes);
    uint64_t cacheLines = approxCacheEntriesNeeded(
        memRefType, memRefShape, tileShape, cacheLineSizeInB);
    cost *= cacheLines * elementsInCacheLine;

    assert(cost != 0 && "Cost should not be zero");
    this->gainCostRatio = gain / cost;
  }

  Optional<int64_t> getFootprint() {
    return this->loop->getMemRefFootprint(this->memRef);
  }

  std::unique_ptr<MemRefRegion> &getMemRefReadRegion() {
    return this->loop->getMemRefReadRegion(this->memRef);
  }

  std::unique_ptr<MemRefRegion> &getMemRefWriteRegion() {
    return this->loop->getMemRefWriteRegion(this->memRef);
  }

  // Check if this packing improves TLB usage of any of its inner loops
  void setTLBImprovement(
      uint64_t l1dtlbPageSizeInKiB, uint64_t l1dtlbEntries,
      SmallMapVector<AffineForOp, LoopInformation, 8> &loopInfoMap,
      ArrayRef<PackingAttributes *> packedTiles = None) {
    this->TLBImprovement = 0;
    uint64_t improvement = 0;

    this->loop->forOp.walk([&](AffineForOp forOp) {
      improvement = this->improvesTLBUsage(
          forOp, l1dtlbPageSizeInKiB, l1dtlbEntries, loopInfoMap, packedTiles);
      // Multiply improvement by the number of times this loop is run
      if (improvement > 0 && forOp != this->loop->forOp) {
        Operation *currOp = forOp;
        while ((currOp = currOp->getParentOp())) {
          if (auto currFor = dyn_cast<AffineForOp>(currOp)) {
            // TODO: give up when trip count has no value
            if (loopInfoMap[currFor].tripCount.hasValue()) {
              improvement *= loopInfoMap[currFor].tripCount.getValue();
              // stop how reaching the target loop for packing
              if (currFor == this->loop->forOp)
                break;
            }
          }
        }
      }
      this->TLBImprovement += improvement;
    });
  }

  /// Approximates TLB usage with and without packing this tile in forOp.
  /// If this computation would not fit L1D TLB before packing
  /// and it fits after packing, return the number of tlb entries packing saves.
  /// Returns 0 otherwise.
  /// PackedTiles defines a list of tiles that should be considered packed.
  /// For them, the packed shape is considered instead of the original memRef
  /// shape. But they are only considered as such if they are packed in the same
  /// or a parent loop.
  uint64_t
  improvesTLBUsage(AffineForOp forOp, uint64_t TLBPageSizeInKiB,
                   uint64_t TLBEntries,
                   SmallMapVector<AffineForOp, LoopInformation, 8> &loopInfoMap,
                   ArrayRef<PackingAttributes *> packedTiles = None) {
    // this check only makes sense if forOp is a child or equal to the packing
    // target loop
    assert(isLoopParentOfOp(this->loop->forOp, forOp) ||
           this->loop->forOp == forOp);
    // Get all memRefs in forOp
    SetVector<Value> memRefs;
    getMemRefsInForOp(forOp, memRefs, /*minRank=*/1);

    if (!memRefs.contains(this->memRef))
      return 0;

    uint64_t entriesPacking = 0;
    uint64_t entriesNoPacking = 0;

    // For every memRef used in this forOp
    for (auto memRef : memRefs) {
      auto memRefType = memRef.getType().cast<MemRefType>();
      // Get memRef shape
      ArrayRef<int64_t> memRefShape = memRefType.getShape();
      // Get tile shape of memRef in forOp
      ArrayRef<int64_t> tileShape =
          loopInfoMap[forOp].getMemRefTileShape(memRef);
      if (tileShape.empty())
        return 0;
      // If this memRef is packed, use packedShape instead of memRef shape
      ArrayRef<int64_t> packedShape;

      // MemRef of this packing cadidate
      if (memRef == this->memRef) {
        packedShape = this->loop->getMemRefTileShape(memRef);
        if (packedShape.empty())
          return 0;
        // If packing does permutation, permute tile shape and packing shape for
        // tlb approximation
        if (this->permutationOrder.hasValue()) {
          SmallVector<int64_t> permutedPackedShape{packedShape.begin(),
                                                   packedShape.end()};
          SmallVector<int64_t> permutedtileShape{tileShape.begin(),
                                                 tileShape.end()};
          permuteArrayBasedOnIndexes(permutedPackedShape,
                                     this->permutationOrder.getValue());
          permuteArrayBasedOnIndexes(permutedtileShape,
                                     this->permutationOrder.getValue());
          entriesPacking += approxCacheEntriesNeeded(
              memRefType, permutedPackedShape, permutedtileShape,
              1024 * TLBPageSizeInKiB);
        } else {
          entriesPacking += approxCacheEntriesNeeded(
              memRefType, packedShape, tileShape, 1024 * TLBPageSizeInKiB);
        }
        entriesNoPacking += approxCacheEntriesNeeded(
            memRefType, memRefShape, tileShape, 1024 * TLBPageSizeInKiB);
        continue;
      }

      int entries = 0;

      // For other memRefs check if they are in the list of packedTiles to do
      // the estimation
      for (auto *packing : packedTiles) {
        if (packing->memRef == memRef &&
            (packing->loop->forOp == forOp ||
             isLoopParentOfOp(packing->loop->forOp, forOp))) {
          packedShape = packing->loop->getMemRefTileShape(memRef);
          if (packedShape.empty())
            return 0;
          if (packing->permutationOrder.hasValue()) {
            SmallVector<int64_t> permutedPackedShape{packedShape.begin(),
                                                     packedShape.end()};
            SmallVector<int64_t> permutedtileShape{tileShape.begin(),
                                                   tileShape.end()};
            permuteArrayBasedOnIndexes(permutedPackedShape,
                                       packing->permutationOrder.getValue());
            permuteArrayBasedOnIndexes(permutedtileShape,
                                       packing->permutationOrder.getValue());
            entries = approxCacheEntriesNeeded(
                              memRefType, permutedPackedShape,
                              permutedtileShape, 1024 * TLBPageSizeInKiB);
          } else {
            entries =
                approxCacheEntriesNeeded(memRefType, packedShape, tileShape,
                                         1024 * TLBPageSizeInKiB);
          }
          break;
        }
      }

      if (entries == 0)
        entries = approxCacheEntriesNeeded(memRefType, memRefShape, tileShape,
                                             1024 * TLBPageSizeInKiB);

      entriesNoPacking += entries;
      entriesPacking += entries;
    }

    if (entriesNoPacking > TLBEntries && entriesPacking <= TLBEntries)
      return entriesNoPacking - entriesPacking;

    return 0;
  }

  bool isContiguous() {
    // Only using read region is needed, write region is may or may not exist
    if (this->getMemRefReadRegion())
      return this->loop->getMemRefIsContiguous(this->memRef);
    return true;
  }

  void setPermutationOrder() {
    auto rank = this->memRef.getType().cast<MemRefType>().getRank();

    this->loop->forOp.walk([&](Operation *op) {
      // get stores and loads related affected by this packing
      if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
        if (this->memRef != loadOp.getMemRef())
          return WalkResult::advance();
        // Give up on non-trivial layout map
        if (!loadOp.getMemRefType().getLayout().isIdentity()) {
          this->permutationOrder = None;
          return WalkResult::interrupt();
        }
      } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
        if (this->memRef != storeOp.getMemRef())
          return WalkResult::advance();
        // Give up on non-trivial layout map
        if (!storeOp.getMemRefType().getLayout().isIdentity()) {
          this->permutationOrder = None;
          return WalkResult::interrupt();
        }
      } else {
        return WalkResult::advance();
      }

      // get access map of load/store
      MemRefAccess access{op};
      AffineValueMap map;
      access.getAccessMap(&map);

      // store max depth of a loopIV owner (a forOp)
      // that is used in each result of a load access map
      SmallVector<uint> indexLoopIVDepth;
      indexLoopIVDepth.reserve(map.getNumResults());

      // for each result of a map expression
      for (unsigned int resultIdx = 0; resultIdx < map.getNumResults();
           resultIdx++) {
        uint maxDepth = 0;
        // for every operand of the map (load/store indices)
        for (size_t operandIdx = 0; operandIdx < map.getNumOperands();
             operandIdx++) {
          auto operand = map.getOperand(operandIdx);
          // if resultIdx^th result is a function of a loop IV and the accesses
          // store max depth
          if (isForInductionVar(operand) &&
              map.isFunctionOf(resultIdx, operand)) {
            AffineForOp ownerForOp = getForInductionVarOwner(operand);
            maxDepth = std::max(maxDepth, getNestingDepth(ownerForOp));
          }
        }
        indexLoopIVDepth.push_back(maxDepth);
      }

      // sort vector of indices idx based on the elements of indexLoopIVDepth
      SmallVector<size_t> idx;
      idx.reserve(rank);
      for (size_t i = 0; i < (size_t)rank; i++)
        idx.push_back(i);

      llvm::stable_sort(idx, [&indexLoopIVDepth](size_t i1, size_t i2) {
        return indexLoopIVDepth[i1] < indexLoopIVDepth[i2];
      });

      // initialize permutation order attribute
      if (!this->permutationOrder.hasValue()) {
        this->permutationOrder = idx;
      } else {
        // if two loads/stores have conflicting permutation orders within this
        // packing, do not permute
        if (this->permutationOrder != idx) {
          this->permutationOrder = None;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    // set permutationOrder to None if no permutation is needed
    SmallVector<size_t> identity;
    identity.reserve(rank);
    for (size_t i = 0; i < (size_t)rank; i++)
      identity.push_back(i);
    if (this->permutationOrder == identity)
      this->permutationOrder = None;
  }

  void setInnermostLoopIVPermutation(uint64_t cacheLineSizeInB) {
    this->innermostLoopIVPermutation = false;

    // Check if there is permutation
    if (!this->permutationOrder.hasValue())
      return;

    this->loop->forOp.walk([&](Operation *op) {
      // get stores and loads related affected by this packing
      if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
        if (this->memRef != loadOp.getMemRef())
          return WalkResult::advance();
        // Give up on non-trivial layout map
        if (!loadOp.getMemRefType().getLayout().isIdentity()) {
          this->permutationOrder = None;
          return WalkResult::interrupt();
        }
      } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
        if (this->memRef != storeOp.getMemRef())
          return WalkResult::advance();
        // Give up on non-trivial layout map
        if (!storeOp.getMemRefType().getLayout().isIdentity()) {
          this->permutationOrder = None;
          return WalkResult::interrupt();
        }
      } else {
        return WalkResult::advance();
      }

      // Get access map of load/store
      MemRefAccess access{op};
      AffineValueMap map;
      access.getAccessMap(&map);

      for (unsigned int resultIdx = 0; resultIdx < map.getNumResults();
           resultIdx++) {
        for (size_t operandIdx = 0; operandIdx < map.getNumOperands();
             operandIdx++) {
          auto operand = map.getOperand(operandIdx);
          if (isForInductionVar(operand) &&
              map.isFunctionOf(resultIdx, operand)) {
            AffineForOp ownerForOp = getForInductionVarOwner(operand);
            if (isInnermostAffineForOp(ownerForOp) &&
                this->permutationOrder.getValue()[resultIdx] != resultIdx) {
              this->innermostLoopIVPermutation = true;
              return WalkResult::interrupt();
            }
          }
        }
      }

      return WalkResult::advance();
    });

    // check if innermost loop is at least two cache lines long
    // otherwise ignore innermost loop permutation
    if (this->innermostLoopIVPermutation) {
      auto packedShape = this->loop->getMemRefTileShape(this->memRef);
      if (packedShape.empty()) {
        this->innermostLoopIVPermutation = false;
        return;
      }
      SmallVector<int64_t> permutedPackedShape{packedShape.begin(),
                                               packedShape.end()};
      permuteArrayBasedOnIndexes(permutedPackedShape,
                                 this->permutationOrder.getValue());
      auto innermostElements = permutedPackedShape.back();
      auto memRefType = this->memRef.getType().cast<MemRefType>();
      int typeSizeBytes = getMemRefEltSizeInBytes(memRefType);
      if (innermostElements * typeSizeBytes < 2 * (int) cacheLineSizeInB)
        this->innermostLoopIVPermutation = false;
    }
  }
};

/// Analyse and apply packing to a loop and its nestings.
void LoopPacking::runOnOuterForOp(AffineForOp outerForOp,
                                  DenseSet<Operation *> &copyNests) {
  LLVM_DEBUG(dbgs() << "\n[DEBUG] Now Running IF Analysis" << "\n");
                                    
  // Skip copy forOps
  if (copyNests.count(outerForOp) != 0)
    return;

  // Holds all memrefs with rank>=2 used in this for op and guarantees order
  SetVector<Value> memRefs;
  getMemRefsInForOp(outerForOp, memRefs);
  if (memRefs.empty()) {
    LLVM_DEBUG(
        dbgs() << "[DEBUG] Not packing, no memRefs with rank >= 2 found.\n");
    return;
  }

  // Collect information of the loops
  SmallMapVector<AffineForOp, LoopInformation, 8> loopInfoMap;
  outerForOp.walk(
      [&](AffineForOp forOp) { loopInfoMap[forOp] = LoopInformation(forOp); });

  // At least three loops are required for an invariant loop
  // to appear in memrefs with rank >= 2
  if (loopInfoMap.size() < 3) {
    LLVM_DEBUG(
        dbgs() << "[DEBUG] Not packing, loop has depth smaller than 3.\n");
    return;
  }

  // Structure storing packing cadidates
  SmallVector<PackingAttributes> packingCadidates;

  // Add packing cadidates if loop is invariant to a memRef
  for (auto &loopInfo : loopInfoMap) {
    for (const auto memRef : memRefs) {
      AffineForOp forOp = loopInfo.first;
      LoopInformation &forOpInfo = loopInfo.second;
      // Skip outermost loop: would pack the entire memRef
      // TODO: consider also depth 0
      if (this->ignoreInvarianceCheck ||
          (forOpInfo.depth != 0 && isMemRefUsedInForOp(forOp, memRef) &&
           isLoopInvariantToMemRef(forOp, memRef))) {
        packingCadidates.push_back(PackingAttributes(memRef, forOpInfo));
      }
    }
  }
  if (packingCadidates.empty()) {
    LLVM_DEBUG(dbgs() << "[DEBUG] Not packing, no invariant loops found.\n");
    return;
  }

  // Check if packing should be permuted
  if (!disablePermutation) {
    for (auto &packing : packingCadidates) {
      packing.setPermutationOrder();
    }
  }

  if (!this->ignoreContiguousCheck) {
    // Remove packing options that are contiguous and have no permutation
    packingCadidates.erase(
        std::remove_if(
            packingCadidates.begin(), packingCadidates.end(),
            [&](PackingAttributes &attr) {
              if (attr.isContiguous() && attr.permutationOrder == None) {
                LLVM_DEBUG(dbgs() << "[OPTION REMOVED] Region is contiguous "
                                     "and has no permutation.\n");
                attr.debug();
                LLVM_DEBUG(dbgs()
                           << "--------------------------------------------\n");
                return true;
              }
              return false;
            }),
        packingCadidates.end());
  }
  if (packingCadidates.empty()) {
    LLVM_DEBUG(dbgs() << "[DEBUG] Not packing, regions are contiguous and have "
                         "no permutation.\n");
    return;
  }

  // Remove packing options that have no read region (only write)
  // or that could not compute memRef read regions.
  // Only writing to a memRef will not benefit from packing.
  packingCadidates.erase(
      std::remove_if(
          packingCadidates.begin(), packingCadidates.end(),
          [&](PackingAttributes &attr) {
            if (!attr.getMemRefReadRegion()) {
              LLVM_DEBUG(dbgs() << "[OPTION REMOVED] Could not compute memRef "
                                   "read regions or has only write region.\n");
              attr.debug();
              LLVM_DEBUG(dbgs()
                         << "--------------------------------------------\n");
              return true;
            }
            return false;
          }),
      packingCadidates.end());
  if (packingCadidates.empty()) {
    LLVM_DEBUG(
        dbgs() << "[DEBUG] Not packing, could not compute memRef regions.\n");
    return;
  }

  // No need for packing if everything already fits in l1 cache
  Optional<int64_t> totalFootprint =
      getMemoryFootprintBytesWithBranches(outerForOp, /*memorySpace=*/0);
  
  LLVM_DEBUG(dbgs() << "[DEBUG] Total footprint: " << totalFootprint.getValueOr(0) << "\n");
    
  if (!this->ignoreCache && totalFootprint.hasValue() &&
      static_cast<uint64_t>(totalFootprint.getValue()) <
          this->l1CacheSizeInKiB * 1024) {
    LLVM_DEBUG(
        dbgs() << "[DEBUG] Not packing, everything already fits in L1.\n");
    return;
  }

  // Set cache threshold
  uint64_t cacheThresholdSizeInKiB = this->l3CacheSizeInKiB;
  // If the computation fits in one cache, consider only upper levels
  if (totalFootprint.hasValue()) {
    uint64_t totalFootprintValue =
        static_cast<uint64_t>(totalFootprint.getValue());
    // Computation fits in L2
    if (totalFootprintValue < this->l2CacheSizeInKiB * 1024) {
      cacheThresholdSizeInKiB = this->l1CacheSizeInKiB;
      LLVM_DEBUG(dbgs() << "[DEBUG] Cache threshold set to L1.\n");
      // Computation fits in L3
    } else if (totalFootprintValue < this->l3CacheSizeInKiB * 1024) {
      cacheThresholdSizeInKiB = this->l2CacheSizeInKiB;
      LLVM_DEBUG(dbgs() << "[DEBUG] Cache threshold set to L2.\n");
      // Computation does not fit in L3
    } else {
      LLVM_DEBUG(dbgs() << "[DEBUG] Cache threshold set to L3.\n");
    }
  }

  // Copy options: {generateDma, slowMemorySpace, fastMemorySpace,
  // tagMemorySpace, fastMemCapacityBytes}
  AffineCopyOptions copyOptions = {false, 0, 0, 0,
                                   cacheThresholdSizeInKiB * 1024};

  // Set packing residency footprint
  for (auto &packing : packingCadidates) {
    packing.setResidencyFootprint();
  }

  // Remove packing options that could not compute footprint or residency
  // footprint
  packingCadidates.erase(
      std::remove_if(
          packingCadidates.begin(), packingCadidates.end(),
          [&](PackingAttributes &attr) {
            auto footprint = attr.getFootprint();
            if (!footprint.hasValue() || !attr.residencyFootprint.hasValue() ||
                (footprint.hasValue() && footprint.getValue() == 0)) {
              LLVM_DEBUG(dbgs() << "[OPTION REMOVED] Could not compute "
                                   "footprint or residency footprint.\n");
              attr.debug();
              LLVM_DEBUG(dbgs()
                         << "--------------------------------------------\n");
              return true;
            }
            return false;
          }),
      packingCadidates.end());
  if (packingCadidates.empty()) {
    LLVM_DEBUG(dbgs() << "[DEBUG] Not packing, could not compute footprint or "
                         "residency footprints.\n");
    return;
  }

  // Remove packing options that have a footprint bigger than the target cache
  // or that are too small
  if (!this->ignoreCache) {
    packingCadidates.erase(
        std::remove_if(
            packingCadidates.begin(), packingCadidates.end(),
            [&](PackingAttributes &attr) {
              auto footprint = attr.getFootprint();
              auto footprintValue = static_cast<uint64_t>(footprint.getValue());
              if (footprintValue >= cacheThresholdSizeInKiB * 1024) {
                LLVM_DEBUG(
                    dbgs()
                    << "[OPTION REMOVED] Footprint bigger than cache.\n");
                attr.debug();
                LLVM_DEBUG(dbgs()
                           << "--------------------------------------------\n");
                return true;
              }
              if (attr.residencyFootprint.getValue() >
                  cacheThresholdSizeInKiB * 1024) {
                LLVM_DEBUG(dbgs() << "[OPTION REMOVED] Residency footprint "
                                     "bigger than target level of cache.\n");
                attr.debug();
                LLVM_DEBUG(dbgs()
                           << "--------------------------------------------\n");
                return true;
              }
              return false;
            }),
        packingCadidates.end());
  }
  if (packingCadidates.empty()) {
    LLVM_DEBUG(
        dbgs() << "[DEBUG] Not packing, options were too big or too small.\n");
    return;
  }

  // Set TLB improvement for each packing
  // and see if their permutation involves the innermost loop IV
  for (auto &packing : packingCadidates) {
    // only possible after creating regions
    packing.setTLBImprovement(this->l1dtlbPageSizeInKiB, this->l1dtlbEntries,
                              loopInfoMap);
    packing.setInnermostLoopIVPermutation(this->cacheLineSizeInB);
  }
  // Remove packing options that do not improve TLB use
  if (!this->ignoreTLB) {
    packingCadidates.erase(
        std::remove_if(
            packingCadidates.begin(), packingCadidates.end(),
            [&](PackingAttributes &attr) {
              // TODO: consider L2 tlb
              // TODO: consider packing combinations to improve TLB
              if (attr.TLBImprovement == 0 &&
                  !attr.innermostLoopIVPermutation) {
                LLVM_DEBUG(dbgs()
                           << "[OPTION REMOVED] Does not improve TLB usage.\n");
                attr.debug();
                LLVM_DEBUG(dbgs()
                           << "--------------------------------------------\n");
                return true;
              }
              return false;
            }),
        packingCadidates.end());
    if (packingCadidates.empty()) {
      LLVM_DEBUG(dbgs() << "[DEBUG] Not packing, does not improve tlb usage "
                           "and have no innermost permutation\n");
      return;
    }
  }

  // Set gain/cost ratio
  for (auto &packing : packingCadidates) {
    // only possible after creating regions
    packing.setGainCostRatio(this->cacheLineSizeInB);
  }

  // Output all possible location and their option ID (counter)
  LLVM_DEBUG(dbgs() << "[ALL LOCATIONS] \n");
  int counter = 1;
  for (auto &packing : packingCadidates) {
    packing.id = counter++;
    packing.debug();
  }
  LLVM_DEBUG(
      dbgs() << "--------------------------------------------------------.\n");

  if (this->packingOptions.empty()) {
    // Greedy approach to select MemRef packings.
    SmallVector<PackingAttributes *> greedyPackings{};

    // Sort the packing options from best to worst options according to the
    // gain/cost.
    std::sort(packingCadidates.begin(), packingCadidates.end(),
              std::greater<PackingAttributes>());

    for (auto &packing : packingCadidates) {
      // add best packing
      if (greedyPackings.empty()) {
        greedyPackings.push_back(&packing);
        continue;
      }

      // Recalculate TLB improvement considering packing that were already
      // selected (greedyPackings)
      packing.setTLBImprovement(this->l1dtlbPageSizeInKiB, this->l1dtlbEntries,
                                loopInfoMap, greedyPackings);
      // Do not pack if no improvement and no permutation on innermost loop iv
      if (packing.TLBImprovement == 0 && !packing.innermostLoopIVPermutation)
        continue;

      bool shouldPack = true;
      for (auto *greedyPacking : greedyPackings) {
        // trying to pack the same memref again:
        // should only pack if packings are independent
        if (greedyPacking->memRef == packing.memRef) {
          if (isLoopParentOfOp(packing.loop->forOp,
                               greedyPacking->loop->forOp) ||
              isLoopParentOfOp(greedyPacking->loop->forOp,
                               packing.loop->forOp)) {
            shouldPack = false;
            break;
          }
        }
      }
      if (shouldPack)
        greedyPackings.push_back(&packing);
    }

    LLVM_DEBUG(dbgs() << "[GREEDILY SELECTED] "
                      << "\n");

    for (auto *packing : greedyPackings)
      packing->debug();

    // For each memref, the packs are generated at the found hoisting locations.
    for (auto *packing : greedyPackings) {
      LLVM_DEBUG(dbgs() << "[DEBUG] Packing memRef: " << packing->memRef
                        << "\n");

      ArrayRef<size_t> permutation{};
      if (packing->permutationOrder.hasValue())
        permutation = packing->permutationOrder.getValue();

      if (failed(generatePackings(packing->memRef, packing->loop->forOp,
                                  packing->getMemRefReadRegion(),
                                  packing->getMemRefWriteRegion(), copyNests,
                                  copyOptions, permutation))) {
        LLVM_DEBUG(dbgs() << "   [DEBUG] Failed to generate packing"
                          << "\n\n");
      } else {
        LLVM_DEBUG(dbgs() << "   [DEBUG] Succeeded generating packing"
                          << "\n\n");
      }
    }
  } else /* User selected*/ {

    // Store user-selected packings
    SmallVector<PackingAttributes *> manualPackings{};

    // Put user-selected packing in a set
    SmallSet<int, 4> options{};
    for (auto opt : this->packingOptions) {
      options.insert(opt);
    }

    // Add user-selected packings to vector
    uint64_t fp = 0;
    counter = 1;
    for (auto &packing : packingCadidates) {
      auto footprint = packing.getFootprint();
      if (options.contains(counter)) {
        manualPackings.push_back(&packing);
        if (footprint.hasValue()) {
          fp += footprint.getValue();
        }
      }
      counter++;
    }

    // For every pair of packings in manual packings
    for (auto *packing1 : manualPackings) {
      for (auto *packing2 : manualPackings) {
        if (packing1 == packing2)
          continue;

        // trying to pack the same memref again:
        // should only pack if packings are independent
        if (packing1->memRef == packing2->memRef) {
          if (isLoopParentOfOp(packing1->loop->forOp, packing2->loop->forOp) ||
              isLoopParentOfOp(packing2->loop->forOp, packing1->loop->forOp)) {
            LLVM_DEBUG(dbgs()
                       << "[ERROR] Same MemRef was selected to be packed twice "
                          "and one packing is a parent of the other"
                       << "\n");
            return signalPassFailure();
          }
        }
      }
    }

    LLVM_DEBUG(dbgs() << "[MANUALLY SELECTED] "
                      << "TOTAL FOOTPRINT: " << fp << "\n");
    for (auto *packing : manualPackings)
      packing->debug();

    for (auto *packing : manualPackings) {
      LLVM_DEBUG(dbgs() << "[DEBUG] Packing memRef: " << packing->memRef
                        << "\n");

      ArrayRef<size_t> permutation{};
      if (packing->permutationOrder.hasValue())
        permutation = packing->permutationOrder.getValue();

      if (failed(generatePackings(packing->memRef, packing->loop->forOp,
                                  packing->getMemRefReadRegion(),
                                  packing->getMemRefWriteRegion(), copyNests,
                                  copyOptions, permutation))) {
        LLVM_DEBUG(dbgs() << "   [DEBUG] Failed to generate packing"
                          << "\n\n");
      } else {
        LLVM_DEBUG(dbgs() << "   [DEBUG] Succeeded generating packing"
                          << "\n\n");
      }
    }
  }
}

void LoopPacking::runOnOperation() {
  func::FuncOp f = getOperation();

  // Add 'noalias' to memRefs
  if (this->addNoAlias) {
    for (BlockArgument argument : f.getArguments()) {
      if (argument.getType().dyn_cast<MemRefType>()) {
        f.setArgAttr(argument.getArgNumber(),
                     mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                     mlir::UnitAttr::get(&getContext()));
      }
    }
  }

  // Add ffast-math to arith and math operations
  if (this->addFFastMath) {
    f.walk([&](Operation *op) {
      if (isa<arith::ArithmeticDialect, math::MathDialect>(op->getDialect())) {
        op->setAttr("fastmathFlags",
                    mlir::LLVM::FMFAttr::get(&getContext(),
                                             mlir::LLVM::FastmathFlags::fast));
      }
    });
  }

  // Disable packing
  if (this->packingOptions.size() == 1 && this->packingOptions[0] == -1)
    return;

  // Store created copies to skip them
  DenseSet<Operation *> copyNests;
  copyNests.clear();

  for (auto &block : f) {
    // Every outer forOp
    for (AffineForOp outerForOp : block.getOps<AffineForOp>())
      runOnOuterForOp(outerForOp, copyNests);
  }
}
