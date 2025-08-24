#map = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<() -> (0)>
#map4 = affine_map<()[s0] -> (s0)>
#map5 = affine_map<() -> (0, 0)>
#map6 = affine_map<() -> (8, 8)>
"builtin.module"() ({
  "func.func"() <{function_type = (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index) -> (), sym_name = "mm_analysis_one_output_per_core"}> ({
  ^bb0(%arg66: memref<?x?xf32>, %arg67: memref<?x?xf32>, %arg68: memref<?x?xf32>, %arg69: index, %arg70: index, %arg71: index):
    %42 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %43 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %44 = "df.interconnects"(%42, %43) <{map = #map}> : (index, index) -> !df.interconnect
    %45 = "df.interconnects"(%42, %43) <{map = #map1}> : (index, index) -> !df.interconnect
    "affine.parallel"(%arg69, %arg70) <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map2}> ({
    ^bb0(%arg72: index, %arg73: index):
      "affine.for"(%arg71) <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 1, 0>, step = 1 : index, upperBoundMap = #map4}> ({
      ^bb0(%arg74: index):
        %46 = "affine.load"(%arg66, %arg72, %arg74) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
        %47 = "affine.load"(%arg67, %arg74, %arg73) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
        %48 = "affine.load"(%arg68, %arg72, %arg73) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
        "affine.store"(%48, %arg68, %arg72, %arg73) <{map = #map2}> : (f32, memref<?x?xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) : (index) -> ()
      "affine.yield"() : () -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index) -> (), sym_name = "mm_analysis_one_output_per_core__plan_p0[8,8]_p1[]"}> ({
  ^bb0(%arg55: memref<?x?xf32>, %arg56: memref<?x?xf32>, %arg57: memref<?x?xf32>, %arg58: index, %arg59: index, %arg60: index):
    %35 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %36 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %37 = "df.interconnects"(%35, %36) <{map = #map}> : (index, index) -> !df.interconnect
    %38 = "df.interconnects"(%35, %36) <{map = #map1}> : (index, index) -> !df.interconnect
    "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map6}> ({
    ^bb0(%arg61: index, %arg62: index):
      "affine.parallel"(%arg58, %arg59) <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map2}> ({
      ^bb0(%arg63: index, %arg64: index):
        "affine.for"(%arg60) <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 1, 0>, step = 1 : index, upperBoundMap = #map4}> ({
        ^bb0(%arg65: index):
          %39 = "affine.load"(%arg55, %arg63, %arg65) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %40 = "affine.load"(%arg56, %arg65, %arg64) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %41 = "affine.load"(%arg57, %arg63, %arg64) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          "affine.store"(%41, %arg57, %arg63, %arg64) <{map = #map2}> : (f32, memref<?x?xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : (index) -> ()
        "affine.yield"() : () -> ()
      }) : (index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index) -> (), sym_name = "mm_analysis_one_output_per_core__plan_p0[8]_p1[8]"}> ({
  ^bb0(%arg44: memref<?x?xf32>, %arg45: memref<?x?xf32>, %arg46: memref<?x?xf32>, %arg47: index, %arg48: index, %arg49: index):
    %28 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %29 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %30 = "df.interconnects"(%28, %29) <{map = #map}> : (index, index) -> !df.interconnect
    %31 = "df.interconnects"(%28, %29) <{map = #map1}> : (index, index) -> !df.interconnect
    "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map6}> ({
    ^bb0(%arg50: index, %arg51: index):
      "affine.parallel"(%arg47, %arg48) <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map2}> ({
      ^bb0(%arg52: index, %arg53: index):
        "affine.for"(%arg49) <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 1, 0>, step = 1 : index, upperBoundMap = #map4}> ({
        ^bb0(%arg54: index):
          %32 = "affine.load"(%arg44, %arg52, %arg54) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %33 = "affine.load"(%arg45, %arg54, %arg53) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %34 = "affine.load"(%arg46, %arg52, %arg53) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          "affine.store"(%34, %arg46, %arg52, %arg53) <{map = #map2}> : (f32, memref<?x?xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : (index) -> ()
        "affine.yield"() : () -> ()
      }) : (index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index) -> (), sym_name = "mm_analysis_one_output_per_core__plan_p0[]_p1[8,8]"}> ({
  ^bb0(%arg33: memref<?x?xf32>, %arg34: memref<?x?xf32>, %arg35: memref<?x?xf32>, %arg36: index, %arg37: index, %arg38: index):
    %21 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %22 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %23 = "df.interconnects"(%21, %22) <{map = #map}> : (index, index) -> !df.interconnect
    %24 = "df.interconnects"(%21, %22) <{map = #map1}> : (index, index) -> !df.interconnect
    "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map6}> ({
    ^bb0(%arg39: index, %arg40: index):
      "affine.parallel"(%arg36, %arg37) <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map2}> ({
      ^bb0(%arg41: index, %arg42: index):
        "affine.for"(%arg38) <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 1, 0>, step = 1 : index, upperBoundMap = #map4}> ({
        ^bb0(%arg43: index):
          %25 = "affine.load"(%arg33, %arg41, %arg43) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %26 = "affine.load"(%arg34, %arg43, %arg42) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %27 = "affine.load"(%arg35, %arg41, %arg42) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          "affine.store"(%27, %arg35, %arg41, %arg42) <{map = #map2}> : (f32, memref<?x?xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : (index) -> ()
        "affine.yield"() : () -> ()
      }) : (index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index) -> (), sym_name = "mm_analysis_one_output_per_core__plan_p0[8,8]_p1[]"}> ({
  ^bb0(%arg22: memref<?x?xf32>, %arg23: memref<?x?xf32>, %arg24: memref<?x?xf32>, %arg25: index, %arg26: index, %arg27: index):
    %14 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %15 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %16 = "df.interconnects"(%14, %15) <{map = #map}> : (index, index) -> !df.interconnect
    %17 = "df.interconnects"(%14, %15) <{map = #map1}> : (index, index) -> !df.interconnect
    "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map6}> ({
    ^bb0(%arg28: index, %arg29: index):
      "affine.parallel"(%arg25, %arg26) <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map2}> ({
      ^bb0(%arg30: index, %arg31: index):
        "affine.for"(%arg27) <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 1, 0>, step = 1 : index, upperBoundMap = #map4}> ({
        ^bb0(%arg32: index):
          %18 = "affine.load"(%arg22, %arg30, %arg32) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %19 = "affine.load"(%arg23, %arg32, %arg31) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %20 = "affine.load"(%arg24, %arg30, %arg31) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          "affine.store"(%20, %arg24, %arg30, %arg31) <{map = #map2}> : (f32, memref<?x?xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : (index) -> ()
        "affine.yield"() : () -> ()
      }) : (index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index) -> (), sym_name = "mm_analysis_one_output_per_core__plan_p0[8]_p1[8]"}> ({
  ^bb0(%arg11: memref<?x?xf32>, %arg12: memref<?x?xf32>, %arg13: memref<?x?xf32>, %arg14: index, %arg15: index, %arg16: index):
    %7 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %8 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %9 = "df.interconnects"(%7, %8) <{map = #map}> : (index, index) -> !df.interconnect
    %10 = "df.interconnects"(%7, %8) <{map = #map1}> : (index, index) -> !df.interconnect
    "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map6}> ({
    ^bb0(%arg17: index, %arg18: index):
      "affine.parallel"(%arg14, %arg15) <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map2}> ({
      ^bb0(%arg19: index, %arg20: index):
        "affine.for"(%arg16) <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 1, 0>, step = 1 : index, upperBoundMap = #map4}> ({
        ^bb0(%arg21: index):
          %11 = "affine.load"(%arg11, %arg19, %arg21) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %12 = "affine.load"(%arg12, %arg21, %arg20) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %13 = "affine.load"(%arg13, %arg19, %arg20) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          "affine.store"(%13, %arg13, %arg19, %arg20) <{map = #map2}> : (f32, memref<?x?xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : (index) -> ()
        "affine.yield"() : () -> ()
      }) : (index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index) -> (), sym_name = "mm_analysis_one_output_per_core__plan_p0[]_p1[8,8]"}> ({
  ^bb0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index):
    %0 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %1 = "df.spatial_dim"() <{size = 8 : i64}> : () -> index
    %2 = "df.interconnects"(%0, %1) <{map = #map}> : (index, index) -> !df.interconnect
    %3 = "df.interconnects"(%0, %1) <{map = #map1}> : (index, index) -> !df.interconnect
    "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map6}> ({
    ^bb0(%arg6: index, %arg7: index):
      "affine.parallel"(%arg3, %arg4) <{lowerBoundsGroups = dense<1> : tensor<2xi32>, lowerBoundsMap = #map5, reductions = [], steps = [1, 1], upperBoundsGroups = dense<1> : tensor<2xi32>, upperBoundsMap = #map2}> ({
      ^bb0(%arg8: index, %arg9: index):
        "affine.for"(%arg5) <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 1, 0>, step = 1 : index, upperBoundMap = #map4}> ({
        ^bb0(%arg10: index):
          %4 = "affine.load"(%arg0, %arg8, %arg10) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %5 = "affine.load"(%arg1, %arg10, %arg9) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          %6 = "affine.load"(%arg2, %arg8, %arg9) <{map = #map2}> : (memref<?x?xf32>, index, index) -> f32
          "affine.store"(%6, %arg2, %arg8, %arg9) <{map = #map2}> : (f32, memref<?x?xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : (index) -> ()
        "affine.yield"() : () -> ()
      }) : (index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

