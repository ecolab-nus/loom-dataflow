module {
  func.func @mm_analysis_one_output_per_core(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    %c8 = arith.constant 8 : index
    %0 = arith.divui %arg3, %c8 : index
    affine.parallel (%arg6, %arg7) = (0, 0) to (symbol(%0), symbol(%arg4)) {
      %c1 = arith.constant 1 : index
      affine.parallel (%arg8) = (0) to (symbol(%c1)) {
        affine.parallel (%arg9) = (0) to (8) {
          affine.for %arg10 = 0 to %arg5 {
            %1 = affine.load %arg0[%arg6, %arg10] : memref<?x?xf32>
            %2 = affine.load %arg1[%arg10, %arg7] : memref<?x?xf32>
            %3 = affine.load %arg2[%arg6, %arg7] : memref<?x?xf32>
            affine.store %3, %arg2[%arg6, %arg7] : memref<?x?xf32>
          }
        }
      }
    }
    return
  }
}

