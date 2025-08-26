#map = affine_map<(d0, d1) -> (d0 * 8 + d1)>
module {
  func.func @mm_analysis_one_output_per_core__d0i0_d1i0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to ((%arg3 ceildiv 8) ceildiv 8, %arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        %0 = affine.apply #map(%arg6, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %1 = affine.apply #map(%0, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %2 = affine.load %arg0[%1, %arg10] : memref<?x?xf32>
            %3 = affine.load %arg1[%arg10, %arg7] : memref<?x?xf32>
            %4 = affine.load %arg2[%1, %arg7] : memref<?x?xf32>
            affine.store %4, %arg2[%1, %arg7] : memref<?x?xf32>
          }
        } {tmd.mapped_to = "x"}
      } {tmd.mapped_to = "y"}
    }
    return
  }
  func.func @mm_analysis_one_output_per_core__d1i0_d0i0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to ((%arg3 ceildiv 8) ceildiv 8, %arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        %0 = affine.apply #map(%arg6, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %1 = affine.apply #map(%0, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %2 = affine.load %arg0[%1, %arg10] : memref<?x?xf32>
            %3 = affine.load %arg1[%arg10, %arg7] : memref<?x?xf32>
            %4 = affine.load %arg2[%1, %arg7] : memref<?x?xf32>
            affine.store %4, %arg2[%1, %arg7] : memref<?x?xf32>
          }
        } {tmd.mapped_to = "y"}
      } {tmd.mapped_to = "x"}
    }
    return
  }
  func.func @mm_analysis_one_output_per_core__d0i0_d1i1(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to (%arg3 ceildiv 8, %arg4 ceildiv 8) {
      affine.parallel (%arg8) = (0) to (8) {
        %0 = affine.apply #map(%arg7, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %1 = affine.apply #map(%arg6, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %2 = affine.load %arg0[%1, %arg10] : memref<?x?xf32>
            %3 = affine.load %arg1[%arg10, %0] : memref<?x?xf32>
            %4 = affine.load %arg2[%1, %0] : memref<?x?xf32>
            affine.store %4, %arg2[%1, %0] : memref<?x?xf32>
          }
        } {tmd.mapped_to = "x"}
      } {tmd.mapped_to = "y"}
    }
    return
  }
  func.func @mm_analysis_one_output_per_core__d1i0_d0i1(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to (%arg3 ceildiv 8, %arg4 ceildiv 8) {
      affine.parallel (%arg8) = (0) to (8) {
        %0 = affine.apply #map(%arg7, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %1 = affine.apply #map(%arg6, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %2 = affine.load %arg0[%1, %arg10] : memref<?x?xf32>
            %3 = affine.load %arg1[%arg10, %0] : memref<?x?xf32>
            %4 = affine.load %arg2[%1, %0] : memref<?x?xf32>
            affine.store %4, %arg2[%1, %0] : memref<?x?xf32>
          }
        } {tmd.mapped_to = "y"}
      } {tmd.mapped_to = "x"}
    }
    return
  }
  func.func @mm_analysis_one_output_per_core__d0i1_d1i1(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to (%arg3, (%arg4 ceildiv 8) ceildiv 8) {
      affine.parallel (%arg8) = (0) to (8) {
        %0 = affine.apply #map(%arg7, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %1 = affine.apply #map(%0, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %2 = affine.load %arg0[%arg6, %arg10] : memref<?x?xf32>
            %3 = affine.load %arg1[%arg10, %1] : memref<?x?xf32>
            %4 = affine.load %arg2[%arg6, %1] : memref<?x?xf32>
            affine.store %4, %arg2[%arg6, %1] : memref<?x?xf32>
          }
        } {tmd.mapped_to = "x"}
      } {tmd.mapped_to = "y"}
    }
    return
  }
  func.func @mm_analysis_one_output_per_core__d1i1_d0i1(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to (%arg3, (%arg4 ceildiv 8) ceildiv 8) {
      affine.parallel (%arg8) = (0) to (8) {
        %0 = affine.apply #map(%arg7, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %1 = affine.apply #map(%0, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %2 = affine.load %arg0[%arg6, %arg10] : memref<?x?xf32>
            %3 = affine.load %arg1[%arg10, %1] : memref<?x?xf32>
            %4 = affine.load %arg2[%arg6, %1] : memref<?x?xf32>
            affine.store %4, %arg2[%arg6, %1] : memref<?x?xf32>
          }
        } {tmd.mapped_to = "y"}
      } {tmd.mapped_to = "x"}
    }
    return
  }
}

