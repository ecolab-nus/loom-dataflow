#map = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map2 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = "df.interconnects"(%0, %1) <{map = #map}> : (index, index) -> !df.interconnect
  %3 = "df.interconnects"(%0, %1) <{map = #map1}> : (index, index) -> !df.interconnect
  func.func @mm_analysis_one_output_per_core__d0i0_d1i0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to ((%arg3 ceildiv 8) ceildiv 8, %arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        %4 = affine.apply #map2(%arg6, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %5 = affine.apply #map2(%4, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %6 = affine.load %arg0[%5, %arg10] : memref<?x?xf32>
            %7 = affine.load %arg1[%arg10, %arg7] : memref<?x?xf32>
            %8 = affine.load %arg2[%5, %arg7] : memref<?x?xf32>
            affine.store %8, %arg2[%5, %arg7] : memref<?x?xf32>
          }
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @mm_analysis_one_output_per_core__d1i0_d0i0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to ((%arg3 ceildiv 8) ceildiv 8, %arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        %4 = affine.apply #map2(%arg6, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %5 = affine.apply #map2(%4, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %6 = affine.load %arg0[%5, %arg10] : memref<?x?xf32>
            %7 = affine.load %arg1[%arg10, %arg7] : memref<?x?xf32>
            %8 = affine.load %arg2[%5, %arg7] : memref<?x?xf32>
            affine.store %8, %arg2[%5, %arg7] : memref<?x?xf32>
          }
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
  func.func @mm_analysis_one_output_per_core__d0i0_d1i1(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to (%arg3 ceildiv 8, %arg4 ceildiv 8) {
      affine.parallel (%arg8) = (0) to (8) {
        %4 = affine.apply #map2(%arg7, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %5 = affine.apply #map2(%arg6, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %6 = affine.load %arg0[%5, %arg10] : memref<?x?xf32>
            %7 = affine.load %arg1[%arg10, %4] : memref<?x?xf32>
            %8 = affine.load %arg2[%5, %4] : memref<?x?xf32>
            affine.store %8, %arg2[%5, %4] : memref<?x?xf32>
          }
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @mm_analysis_one_output_per_core__d1i0_d0i1(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to (%arg3 ceildiv 8, %arg4 ceildiv 8) {
      affine.parallel (%arg8) = (0) to (8) {
        %4 = affine.apply #map2(%arg7, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %5 = affine.apply #map2(%arg6, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %6 = affine.load %arg0[%5, %arg10] : memref<?x?xf32>
            %7 = affine.load %arg1[%arg10, %4] : memref<?x?xf32>
            %8 = affine.load %arg2[%5, %4] : memref<?x?xf32>
            affine.store %8, %arg2[%5, %4] : memref<?x?xf32>
          }
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
  func.func @mm_analysis_one_output_per_core__d0i1_d1i1(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to (%arg3, (%arg4 ceildiv 8) ceildiv 8) {
      affine.parallel (%arg8) = (0) to (8) {
        %4 = affine.apply #map2(%arg7, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %5 = affine.apply #map2(%4, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %6 = affine.load %arg0[%arg6, %arg10] : memref<?x?xf32>
            %7 = affine.load %arg1[%arg10, %5] : memref<?x?xf32>
            %8 = affine.load %arg2[%arg6, %5] : memref<?x?xf32>
            affine.store %8, %arg2[%arg6, %5] : memref<?x?xf32>
          }
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @mm_analysis_one_output_per_core__d1i1_d0i1(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to (%arg3, (%arg4 ceildiv 8) ceildiv 8) {
      affine.parallel (%arg8) = (0) to (8) {
        %4 = affine.apply #map2(%arg7, %arg8)
        affine.parallel (%arg9) = (0) to (8) {
          %5 = affine.apply #map2(%4, %arg9)
          affine.for %arg10 = 0 to %arg5 {
            %6 = affine.load %arg0[%arg6, %arg10] : memref<?x?xf32>
            %7 = affine.load %arg1[%arg10, %5] : memref<?x?xf32>
            %8 = affine.load %arg2[%arg6, %5] : memref<?x?xf32>
            affine.store %8, %arg2[%arg6, %5] : memref<?x?xf32>
          }
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
}

