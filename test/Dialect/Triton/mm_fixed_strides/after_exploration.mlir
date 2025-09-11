#map = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map2 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map3 = affine_map<()[s0] -> ((s0 + 31) floordiv 32)>
#map4 = affine_map<(d0, d1) -> (d1 * 32768 + d0 * 32)>
#map5 = affine_map<(d0, d1) -> (d1 * 16384 + d0 * 64)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map9 = affine_map<(d0, d1) -> (d0, d1)>
#map10 = affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>
#map11 = affine_map<() -> (0)>
#map12 = affine_map<() -> (8)>
#map13 = affine_map<(d0, d1) -> (d1)>
#map14 = affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>
#map15 = affine_map<(d0, d1) -> (d1 ceildiv 8)>
#map16 = affine_map<(d0, d1) -> (d0 ceildiv 8)>
#map17 = affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>
#map18 = affine_map<(d0, d1) -> (d0)>
"builtin.module"() ({
  %0 = "df.spatial_dim"() <{name = "x", size = 8 : i64}> : () -> index
  %1 = "df.spatial_dim"() <{name = "y", size = 8 : i64}> : () -> index
  %2 = "df.interconnects"(%0, %1) <{map = #map}> : (index, index) -> !df.interconnect
  %3 = "df.interconnects"(%0, %1) <{map = #map1}> : (index, index) -> !df.interconnect
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d0i0_d1i0_f0_f1"}> ({
  ^bb0(%arg253: memref<*xf32>, %arg254: memref<*xf32>, %arg255: memref<*xf32>, %arg256: index, %arg257: index, %arg258: index, %arg259: index, %arg260: index, %arg261: index):
    "affine.for"(%arg259, %arg260) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map14}> ({
    ^bb0(%arg262: index):
      "affine.for"(%arg259, %arg260) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map13}> ({
      ^bb0(%arg263: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg264: index):
          %268 = "affine.apply"(%arg262, %arg264) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg265: index):
            %269 = "affine.apply"(%268, %arg265) <{map = #map2}> : (index, index) -> index
            %270 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %271 = "tensor.empty"() : () -> tensor<64x64xf32>
            %272 = "linalg.fill"(%270, %271) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg274: f32, %arg275: f32):
              "linalg.yield"(%arg274) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %273 = "affine.apply"(%arg258) <{map = #map3}> : (index) -> index
            %274 = "arith.constant"() <{value = 0 : index}> : () -> index
            %275 = "arith.constant"() <{value = 1 : index}> : () -> index
            %276 = "scf.for"(%274, %273, %275, %272) ({
            ^bb0(%arg266: index, %arg267: tensor<64x64xf32>):
              %279 = "affine.apply"(%arg266, %269) <{map = #map4}> : (index, index) -> index
              %280 = "memref.reinterpret_cast"(%arg253, %279) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %281 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%280, %281) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %282 = "bufferization.to_tensor"(%281) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %283 = "affine.apply"(%arg263, %arg266) <{map = #map5}> : (index, index) -> index
              %284 = "memref.reinterpret_cast"(%arg254, %283) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %285 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%284, %285) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %286 = "bufferization.to_tensor"(%285) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %287 = "linalg.matmul"(%282, %286, %272) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg271: f32, %arg272: f32, %arg273: f32):
                %290 = "arith.mulf"(%arg271, %arg272) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %291 = "arith.addf"(%arg273, %290) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%291) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %288 = "linalg.generic"(%arg267, %287, %arg267) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg268: f32, %arg269: f32, %arg270: f32):
                %289 = "arith.addf"(%arg268, %arg269) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%289) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%288) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %277 = "affine.apply"(%arg263, %269) <{map = #map10}> : (index, index) -> index
            %278 = "memref.reinterpret_cast"(%arg255, %277) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%276, %278) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "x"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "y"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d0i0_d1i0_f1_f0"}> ({
  ^bb0(%arg230: memref<*xf32>, %arg231: memref<*xf32>, %arg232: memref<*xf32>, %arg233: index, %arg234: index, %arg235: index, %arg236: index, %arg237: index, %arg238: index):
    "affine.for"(%arg236, %arg237) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map13}> ({
    ^bb0(%arg239: index):
      "affine.for"(%arg236, %arg237) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map14}> ({
      ^bb0(%arg240: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg241: index):
          %244 = "affine.apply"(%arg240, %arg241) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg242: index):
            %245 = "affine.apply"(%244, %arg242) <{map = #map2}> : (index, index) -> index
            %246 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %247 = "tensor.empty"() : () -> tensor<64x64xf32>
            %248 = "linalg.fill"(%246, %247) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg251: f32, %arg252: f32):
              "linalg.yield"(%arg251) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %249 = "affine.apply"(%arg235) <{map = #map3}> : (index) -> index
            %250 = "arith.constant"() <{value = 0 : index}> : () -> index
            %251 = "arith.constant"() <{value = 1 : index}> : () -> index
            %252 = "scf.for"(%250, %249, %251, %248) ({
            ^bb0(%arg243: index, %arg244: tensor<64x64xf32>):
              %255 = "affine.apply"(%arg243, %245) <{map = #map4}> : (index, index) -> index
              %256 = "memref.reinterpret_cast"(%arg230, %255) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %257 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%256, %257) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %258 = "bufferization.to_tensor"(%257) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %259 = "affine.apply"(%arg239, %arg243) <{map = #map5}> : (index, index) -> index
              %260 = "memref.reinterpret_cast"(%arg231, %259) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %261 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%260, %261) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %262 = "bufferization.to_tensor"(%261) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %263 = "linalg.matmul"(%258, %262, %248) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg248: f32, %arg249: f32, %arg250: f32):
                %266 = "arith.mulf"(%arg248, %arg249) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %267 = "arith.addf"(%arg250, %266) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%267) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %264 = "linalg.generic"(%arg244, %263, %arg244) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg245: f32, %arg246: f32, %arg247: f32):
                %265 = "arith.addf"(%arg245, %arg246) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%265) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%264) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %253 = "affine.apply"(%arg239, %245) <{map = #map10}> : (index, index) -> index
            %254 = "memref.reinterpret_cast"(%arg232, %253) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%252, %254) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "x"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "y"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d1i0_d0i0_f0_f1"}> ({
  ^bb0(%arg207: memref<*xf32>, %arg208: memref<*xf32>, %arg209: memref<*xf32>, %arg210: index, %arg211: index, %arg212: index, %arg213: index, %arg214: index, %arg215: index):
    "affine.for"(%arg213, %arg214) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map14}> ({
    ^bb0(%arg216: index):
      "affine.for"(%arg213, %arg214) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map13}> ({
      ^bb0(%arg217: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg218: index):
          %220 = "affine.apply"(%arg216, %arg218) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg219: index):
            %221 = "affine.apply"(%220, %arg219) <{map = #map2}> : (index, index) -> index
            %222 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %223 = "tensor.empty"() : () -> tensor<64x64xf32>
            %224 = "linalg.fill"(%222, %223) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg228: f32, %arg229: f32):
              "linalg.yield"(%arg228) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %225 = "affine.apply"(%arg212) <{map = #map3}> : (index) -> index
            %226 = "arith.constant"() <{value = 0 : index}> : () -> index
            %227 = "arith.constant"() <{value = 1 : index}> : () -> index
            %228 = "scf.for"(%226, %225, %227, %224) ({
            ^bb0(%arg220: index, %arg221: tensor<64x64xf32>):
              %231 = "affine.apply"(%arg220, %221) <{map = #map4}> : (index, index) -> index
              %232 = "memref.reinterpret_cast"(%arg207, %231) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %233 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%232, %233) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %234 = "bufferization.to_tensor"(%233) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %235 = "affine.apply"(%arg217, %arg220) <{map = #map5}> : (index, index) -> index
              %236 = "memref.reinterpret_cast"(%arg208, %235) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %237 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%236, %237) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %238 = "bufferization.to_tensor"(%237) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %239 = "linalg.matmul"(%234, %238, %224) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg225: f32, %arg226: f32, %arg227: f32):
                %242 = "arith.mulf"(%arg225, %arg226) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %243 = "arith.addf"(%arg227, %242) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%243) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %240 = "linalg.generic"(%arg221, %239, %arg221) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg222: f32, %arg223: f32, %arg224: f32):
                %241 = "arith.addf"(%arg222, %arg223) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%241) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%240) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %229 = "affine.apply"(%arg217, %221) <{map = #map10}> : (index, index) -> index
            %230 = "memref.reinterpret_cast"(%arg209, %229) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%228, %230) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "y"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "x"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d1i0_d0i0_f1_f0"}> ({
  ^bb0(%arg184: memref<*xf32>, %arg185: memref<*xf32>, %arg186: memref<*xf32>, %arg187: index, %arg188: index, %arg189: index, %arg190: index, %arg191: index, %arg192: index):
    "affine.for"(%arg190, %arg191) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map13}> ({
    ^bb0(%arg193: index):
      "affine.for"(%arg190, %arg191) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map14}> ({
      ^bb0(%arg194: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg195: index):
          %196 = "affine.apply"(%arg194, %arg195) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg196: index):
            %197 = "affine.apply"(%196, %arg196) <{map = #map2}> : (index, index) -> index
            %198 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %199 = "tensor.empty"() : () -> tensor<64x64xf32>
            %200 = "linalg.fill"(%198, %199) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg205: f32, %arg206: f32):
              "linalg.yield"(%arg205) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %201 = "affine.apply"(%arg189) <{map = #map3}> : (index) -> index
            %202 = "arith.constant"() <{value = 0 : index}> : () -> index
            %203 = "arith.constant"() <{value = 1 : index}> : () -> index
            %204 = "scf.for"(%202, %201, %203, %200) ({
            ^bb0(%arg197: index, %arg198: tensor<64x64xf32>):
              %207 = "affine.apply"(%arg197, %197) <{map = #map4}> : (index, index) -> index
              %208 = "memref.reinterpret_cast"(%arg184, %207) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %209 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%208, %209) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %210 = "bufferization.to_tensor"(%209) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %211 = "affine.apply"(%arg193, %arg197) <{map = #map5}> : (index, index) -> index
              %212 = "memref.reinterpret_cast"(%arg185, %211) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %213 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%212, %213) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %214 = "bufferization.to_tensor"(%213) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %215 = "linalg.matmul"(%210, %214, %200) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg202: f32, %arg203: f32, %arg204: f32):
                %218 = "arith.mulf"(%arg202, %arg203) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %219 = "arith.addf"(%arg204, %218) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%219) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %216 = "linalg.generic"(%arg198, %215, %arg198) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg199: f32, %arg200: f32, %arg201: f32):
                %217 = "arith.addf"(%arg199, %arg200) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%217) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%216) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %205 = "affine.apply"(%arg193, %197) <{map = #map10}> : (index, index) -> index
            %206 = "memref.reinterpret_cast"(%arg186, %205) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%204, %206) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "y"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "x"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d0i0_d1i1_f0_f1"}> ({
  ^bb0(%arg161: memref<*xf32>, %arg162: memref<*xf32>, %arg163: memref<*xf32>, %arg164: index, %arg165: index, %arg166: index, %arg167: index, %arg168: index, %arg169: index):
    "affine.for"(%arg167, %arg168) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map16}> ({
    ^bb0(%arg170: index):
      "affine.for"(%arg167, %arg168) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map15}> ({
      ^bb0(%arg171: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg172: index):
          %172 = "affine.apply"(%arg171, %arg172) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg173: index):
            %173 = "affine.apply"(%arg170, %arg173) <{map = #map2}> : (index, index) -> index
            %174 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %175 = "tensor.empty"() : () -> tensor<64x64xf32>
            %176 = "linalg.fill"(%174, %175) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg182: f32, %arg183: f32):
              "linalg.yield"(%arg182) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %177 = "affine.apply"(%arg166) <{map = #map3}> : (index) -> index
            %178 = "arith.constant"() <{value = 0 : index}> : () -> index
            %179 = "arith.constant"() <{value = 1 : index}> : () -> index
            %180 = "scf.for"(%178, %177, %179, %176) ({
            ^bb0(%arg174: index, %arg175: tensor<64x64xf32>):
              %183 = "affine.apply"(%arg174, %173) <{map = #map4}> : (index, index) -> index
              %184 = "memref.reinterpret_cast"(%arg161, %183) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %185 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%184, %185) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %186 = "bufferization.to_tensor"(%185) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %187 = "affine.apply"(%172, %arg174) <{map = #map5}> : (index, index) -> index
              %188 = "memref.reinterpret_cast"(%arg162, %187) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %189 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%188, %189) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %190 = "bufferization.to_tensor"(%189) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %191 = "linalg.matmul"(%186, %190, %176) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg179: f32, %arg180: f32, %arg181: f32):
                %194 = "arith.mulf"(%arg179, %arg180) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %195 = "arith.addf"(%arg181, %194) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%195) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %192 = "linalg.generic"(%arg175, %191, %arg175) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg176: f32, %arg177: f32, %arg178: f32):
                %193 = "arith.addf"(%arg176, %arg177) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%193) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%192) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %181 = "affine.apply"(%172, %173) <{map = #map10}> : (index, index) -> index
            %182 = "memref.reinterpret_cast"(%arg163, %181) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%180, %182) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "x"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "y"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d0i0_d1i1_f1_f0"}> ({
  ^bb0(%arg138: memref<*xf32>, %arg139: memref<*xf32>, %arg140: memref<*xf32>, %arg141: index, %arg142: index, %arg143: index, %arg144: index, %arg145: index, %arg146: index):
    "affine.for"(%arg144, %arg145) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map15}> ({
    ^bb0(%arg147: index):
      "affine.for"(%arg144, %arg145) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map16}> ({
      ^bb0(%arg148: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg149: index):
          %148 = "affine.apply"(%arg147, %arg149) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg150: index):
            %149 = "affine.apply"(%arg148, %arg150) <{map = #map2}> : (index, index) -> index
            %150 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %151 = "tensor.empty"() : () -> tensor<64x64xf32>
            %152 = "linalg.fill"(%150, %151) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg159: f32, %arg160: f32):
              "linalg.yield"(%arg159) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %153 = "affine.apply"(%arg143) <{map = #map3}> : (index) -> index
            %154 = "arith.constant"() <{value = 0 : index}> : () -> index
            %155 = "arith.constant"() <{value = 1 : index}> : () -> index
            %156 = "scf.for"(%154, %153, %155, %152) ({
            ^bb0(%arg151: index, %arg152: tensor<64x64xf32>):
              %159 = "affine.apply"(%arg151, %149) <{map = #map4}> : (index, index) -> index
              %160 = "memref.reinterpret_cast"(%arg138, %159) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %161 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%160, %161) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %162 = "bufferization.to_tensor"(%161) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %163 = "affine.apply"(%148, %arg151) <{map = #map5}> : (index, index) -> index
              %164 = "memref.reinterpret_cast"(%arg139, %163) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %165 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%164, %165) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %166 = "bufferization.to_tensor"(%165) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %167 = "linalg.matmul"(%162, %166, %152) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg156: f32, %arg157: f32, %arg158: f32):
                %170 = "arith.mulf"(%arg156, %arg157) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %171 = "arith.addf"(%arg158, %170) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%171) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %168 = "linalg.generic"(%arg152, %167, %arg152) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg153: f32, %arg154: f32, %arg155: f32):
                %169 = "arith.addf"(%arg153, %arg154) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%169) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%168) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %157 = "affine.apply"(%148, %149) <{map = #map10}> : (index, index) -> index
            %158 = "memref.reinterpret_cast"(%arg140, %157) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%156, %158) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "x"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "y"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d1i0_d0i1_f0_f1"}> ({
  ^bb0(%arg115: memref<*xf32>, %arg116: memref<*xf32>, %arg117: memref<*xf32>, %arg118: index, %arg119: index, %arg120: index, %arg121: index, %arg122: index, %arg123: index):
    "affine.for"(%arg121, %arg122) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map16}> ({
    ^bb0(%arg124: index):
      "affine.for"(%arg121, %arg122) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map15}> ({
      ^bb0(%arg125: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg126: index):
          %124 = "affine.apply"(%arg125, %arg126) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg127: index):
            %125 = "affine.apply"(%arg124, %arg127) <{map = #map2}> : (index, index) -> index
            %126 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %127 = "tensor.empty"() : () -> tensor<64x64xf32>
            %128 = "linalg.fill"(%126, %127) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg136: f32, %arg137: f32):
              "linalg.yield"(%arg136) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %129 = "affine.apply"(%arg120) <{map = #map3}> : (index) -> index
            %130 = "arith.constant"() <{value = 0 : index}> : () -> index
            %131 = "arith.constant"() <{value = 1 : index}> : () -> index
            %132 = "scf.for"(%130, %129, %131, %128) ({
            ^bb0(%arg128: index, %arg129: tensor<64x64xf32>):
              %135 = "affine.apply"(%arg128, %125) <{map = #map4}> : (index, index) -> index
              %136 = "memref.reinterpret_cast"(%arg115, %135) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %137 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%136, %137) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %138 = "bufferization.to_tensor"(%137) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %139 = "affine.apply"(%124, %arg128) <{map = #map5}> : (index, index) -> index
              %140 = "memref.reinterpret_cast"(%arg116, %139) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %141 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%140, %141) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %142 = "bufferization.to_tensor"(%141) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %143 = "linalg.matmul"(%138, %142, %128) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg133: f32, %arg134: f32, %arg135: f32):
                %146 = "arith.mulf"(%arg133, %arg134) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %147 = "arith.addf"(%arg135, %146) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%147) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %144 = "linalg.generic"(%arg129, %143, %arg129) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg130: f32, %arg131: f32, %arg132: f32):
                %145 = "arith.addf"(%arg130, %arg131) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%145) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%144) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %133 = "affine.apply"(%124, %125) <{map = #map10}> : (index, index) -> index
            %134 = "memref.reinterpret_cast"(%arg117, %133) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%132, %134) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "y"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "x"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d1i0_d0i1_f1_f0"}> ({
  ^bb0(%arg92: memref<*xf32>, %arg93: memref<*xf32>, %arg94: memref<*xf32>, %arg95: index, %arg96: index, %arg97: index, %arg98: index, %arg99: index, %arg100: index):
    "affine.for"(%arg98, %arg99) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map15}> ({
    ^bb0(%arg101: index):
      "affine.for"(%arg98, %arg99) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map16}> ({
      ^bb0(%arg102: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg103: index):
          %100 = "affine.apply"(%arg101, %arg103) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg104: index):
            %101 = "affine.apply"(%arg102, %arg104) <{map = #map2}> : (index, index) -> index
            %102 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %103 = "tensor.empty"() : () -> tensor<64x64xf32>
            %104 = "linalg.fill"(%102, %103) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg113: f32, %arg114: f32):
              "linalg.yield"(%arg113) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %105 = "affine.apply"(%arg97) <{map = #map3}> : (index) -> index
            %106 = "arith.constant"() <{value = 0 : index}> : () -> index
            %107 = "arith.constant"() <{value = 1 : index}> : () -> index
            %108 = "scf.for"(%106, %105, %107, %104) ({
            ^bb0(%arg105: index, %arg106: tensor<64x64xf32>):
              %111 = "affine.apply"(%arg105, %101) <{map = #map4}> : (index, index) -> index
              %112 = "memref.reinterpret_cast"(%arg92, %111) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %113 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%112, %113) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %114 = "bufferization.to_tensor"(%113) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %115 = "affine.apply"(%100, %arg105) <{map = #map5}> : (index, index) -> index
              %116 = "memref.reinterpret_cast"(%arg93, %115) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %117 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%116, %117) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %118 = "bufferization.to_tensor"(%117) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %119 = "linalg.matmul"(%114, %118, %104) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg110: f32, %arg111: f32, %arg112: f32):
                %122 = "arith.mulf"(%arg110, %arg111) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %123 = "arith.addf"(%arg112, %122) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%123) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %120 = "linalg.generic"(%arg106, %119, %arg106) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg107: f32, %arg108: f32, %arg109: f32):
                %121 = "arith.addf"(%arg107, %arg108) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%121) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%120) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %109 = "affine.apply"(%100, %101) <{map = #map10}> : (index, index) -> index
            %110 = "memref.reinterpret_cast"(%arg94, %109) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%108, %110) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "y"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "x"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d0i1_d1i1_f0_f1"}> ({
  ^bb0(%arg69: memref<*xf32>, %arg70: memref<*xf32>, %arg71: memref<*xf32>, %arg72: index, %arg73: index, %arg74: index, %arg75: index, %arg76: index, %arg77: index):
    "affine.for"(%arg75, %arg76) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map18}> ({
    ^bb0(%arg78: index):
      "affine.for"(%arg75, %arg76) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map17}> ({
      ^bb0(%arg79: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg80: index):
          %76 = "affine.apply"(%arg79, %arg80) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg81: index):
            %77 = "affine.apply"(%76, %arg81) <{map = #map2}> : (index, index) -> index
            %78 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %79 = "tensor.empty"() : () -> tensor<64x64xf32>
            %80 = "linalg.fill"(%78, %79) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg90: f32, %arg91: f32):
              "linalg.yield"(%arg90) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %81 = "affine.apply"(%arg74) <{map = #map3}> : (index) -> index
            %82 = "arith.constant"() <{value = 0 : index}> : () -> index
            %83 = "arith.constant"() <{value = 1 : index}> : () -> index
            %84 = "scf.for"(%82, %81, %83, %80) ({
            ^bb0(%arg82: index, %arg83: tensor<64x64xf32>):
              %87 = "affine.apply"(%arg82, %arg78) <{map = #map4}> : (index, index) -> index
              %88 = "memref.reinterpret_cast"(%arg69, %87) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %89 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%88, %89) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %90 = "bufferization.to_tensor"(%89) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %91 = "affine.apply"(%77, %arg82) <{map = #map5}> : (index, index) -> index
              %92 = "memref.reinterpret_cast"(%arg70, %91) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %93 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%92, %93) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %94 = "bufferization.to_tensor"(%93) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %95 = "linalg.matmul"(%90, %94, %80) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg87: f32, %arg88: f32, %arg89: f32):
                %98 = "arith.mulf"(%arg87, %arg88) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %99 = "arith.addf"(%arg89, %98) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%99) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %96 = "linalg.generic"(%arg83, %95, %arg83) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg84: f32, %arg85: f32, %arg86: f32):
                %97 = "arith.addf"(%arg84, %arg85) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%97) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%96) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %85 = "affine.apply"(%77, %arg78) <{map = #map10}> : (index, index) -> index
            %86 = "memref.reinterpret_cast"(%arg71, %85) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%84, %86) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "x"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "y"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d0i1_d1i1_f1_f0"}> ({
  ^bb0(%arg46: memref<*xf32>, %arg47: memref<*xf32>, %arg48: memref<*xf32>, %arg49: index, %arg50: index, %arg51: index, %arg52: index, %arg53: index, %arg54: index):
    "affine.for"(%arg52, %arg53) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map17}> ({
    ^bb0(%arg55: index):
      "affine.for"(%arg52, %arg53) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map18}> ({
      ^bb0(%arg56: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg57: index):
          %52 = "affine.apply"(%arg55, %arg57) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg58: index):
            %53 = "affine.apply"(%52, %arg58) <{map = #map2}> : (index, index) -> index
            %54 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %55 = "tensor.empty"() : () -> tensor<64x64xf32>
            %56 = "linalg.fill"(%54, %55) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg67: f32, %arg68: f32):
              "linalg.yield"(%arg67) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %57 = "affine.apply"(%arg51) <{map = #map3}> : (index) -> index
            %58 = "arith.constant"() <{value = 0 : index}> : () -> index
            %59 = "arith.constant"() <{value = 1 : index}> : () -> index
            %60 = "scf.for"(%58, %57, %59, %56) ({
            ^bb0(%arg59: index, %arg60: tensor<64x64xf32>):
              %63 = "affine.apply"(%arg59, %arg56) <{map = #map4}> : (index, index) -> index
              %64 = "memref.reinterpret_cast"(%arg46, %63) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %65 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%64, %65) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %66 = "bufferization.to_tensor"(%65) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %67 = "affine.apply"(%53, %arg59) <{map = #map5}> : (index, index) -> index
              %68 = "memref.reinterpret_cast"(%arg47, %67) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %69 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%68, %69) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %70 = "bufferization.to_tensor"(%69) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %71 = "linalg.matmul"(%66, %70, %56) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg64: f32, %arg65: f32, %arg66: f32):
                %74 = "arith.mulf"(%arg64, %arg65) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %75 = "arith.addf"(%arg66, %74) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%75) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %72 = "linalg.generic"(%arg60, %71, %arg60) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg61: f32, %arg62: f32, %arg63: f32):
                %73 = "arith.addf"(%arg61, %arg62) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%73) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%72) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %61 = "affine.apply"(%53, %arg56) <{map = #map10}> : (index, index) -> index
            %62 = "memref.reinterpret_cast"(%arg48, %61) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%60, %62) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "x"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "y"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d1i1_d0i1_f0_f1"}> ({
  ^bb0(%arg23: memref<*xf32>, %arg24: memref<*xf32>, %arg25: memref<*xf32>, %arg26: index, %arg27: index, %arg28: index, %arg29: index, %arg30: index, %arg31: index):
    "affine.for"(%arg29, %arg30) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map18}> ({
    ^bb0(%arg32: index):
      "affine.for"(%arg29, %arg30) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map17}> ({
      ^bb0(%arg33: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg34: index):
          %28 = "affine.apply"(%arg33, %arg34) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg35: index):
            %29 = "affine.apply"(%28, %arg35) <{map = #map2}> : (index, index) -> index
            %30 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %31 = "tensor.empty"() : () -> tensor<64x64xf32>
            %32 = "linalg.fill"(%30, %31) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg44: f32, %arg45: f32):
              "linalg.yield"(%arg44) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %33 = "affine.apply"(%arg28) <{map = #map3}> : (index) -> index
            %34 = "arith.constant"() <{value = 0 : index}> : () -> index
            %35 = "arith.constant"() <{value = 1 : index}> : () -> index
            %36 = "scf.for"(%34, %33, %35, %32) ({
            ^bb0(%arg36: index, %arg37: tensor<64x64xf32>):
              %39 = "affine.apply"(%arg36, %arg32) <{map = #map4}> : (index, index) -> index
              %40 = "memref.reinterpret_cast"(%arg23, %39) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %41 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%40, %41) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %42 = "bufferization.to_tensor"(%41) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %43 = "affine.apply"(%29, %arg36) <{map = #map5}> : (index, index) -> index
              %44 = "memref.reinterpret_cast"(%arg24, %43) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %45 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%44, %45) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %46 = "bufferization.to_tensor"(%45) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %47 = "linalg.matmul"(%42, %46, %32) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg41: f32, %arg42: f32, %arg43: f32):
                %50 = "arith.mulf"(%arg41, %arg42) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %51 = "arith.addf"(%arg43, %50) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%51) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %48 = "linalg.generic"(%arg37, %47, %arg37) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg38: f32, %arg39: f32, %arg40: f32):
                %49 = "arith.addf"(%arg38, %arg39) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%49) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%48) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %37 = "affine.apply"(%29, %arg32) <{map = #map10}> : (index, index) -> index
            %38 = "memref.reinterpret_cast"(%arg25, %37) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%36, %38) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "y"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "x"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index, index, index, index) -> (), sym_name = "matmul_kernel__d1i1_d0i1_f1_f0"}> ({
  ^bb0(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index):
    "affine.for"(%arg6, %arg7) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map17}> ({
    ^bb0(%arg9: index):
      "affine.for"(%arg6, %arg7) <{lowerBoundMap = #map11, operandSegmentSizes = array<i32: 0, 2, 0>, step = 1 : index, upperBoundMap = #map18}> ({
      ^bb0(%arg10: index):
        "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
        ^bb0(%arg11: index):
          %4 = "affine.apply"(%arg9, %arg11) <{map = #map2}> : (index, index) -> index
          "affine.parallel"() <{lowerBoundsGroups = dense<1> : tensor<1xi32>, lowerBoundsMap = #map11, reductions = [], steps = [1], upperBoundsGroups = dense<1> : tensor<1xi32>, upperBoundsMap = #map12}> ({
          ^bb0(%arg12: index):
            %5 = "affine.apply"(%4, %arg12) <{map = #map2}> : (index, index) -> index
            %6 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
            %7 = "tensor.empty"() : () -> tensor<64x64xf32>
            %8 = "linalg.fill"(%6, %7) <{operandSegmentSizes = array<i32: 1, 1>}> ({
            ^bb0(%arg21: f32, %arg22: f32):
              "linalg.yield"(%arg21) : (f32) -> ()
            }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
            %9 = "affine.apply"(%arg5) <{map = #map3}> : (index) -> index
            %10 = "arith.constant"() <{value = 0 : index}> : () -> index
            %11 = "arith.constant"() <{value = 1 : index}> : () -> index
            %12 = "scf.for"(%10, %9, %11, %8) ({
            ^bb0(%arg13: index, %arg14: tensor<64x64xf32>):
              %15 = "affine.apply"(%arg13, %arg10) <{map = #map4}> : (index, index) -> index
              %16 = "memref.reinterpret_cast"(%arg0, %15) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 32>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x32xf32, strided<[512, 1], offset: ?>>
              %17 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x32xf32>
              "memref.copy"(%16, %17) : (memref<64x32xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32>) -> ()
              %18 = "bufferization.to_tensor"(%17) <{restrict, writable}> : (memref<64x32xf32>) -> tensor<64x32xf32>
              %19 = "affine.apply"(%5, %arg13) <{map = #map5}> : (index, index) -> index
              %20 = "memref.reinterpret_cast"(%arg1, %19) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<32x64xf32, strided<[512, 1], offset: ?>>
              %21 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32>
              "memref.copy"(%20, %21) : (memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
              %22 = "bufferization.to_tensor"(%21) <{restrict, writable}> : (memref<32x64xf32>) -> tensor<32x64xf32>
              %23 = "linalg.matmul"(%18, %22, %8) <{indexing_maps = [#map6, #map7, #map8], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg18: f32, %arg19: f32, %arg20: f32):
                %26 = "arith.mulf"(%arg18, %arg19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                %27 = "arith.addf"(%arg20, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%27) : (f32) -> ()
              }) : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              %24 = "linalg.generic"(%arg14, %23, %arg14) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):
                %25 = "arith.addf"(%arg15, %arg16) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
                "linalg.yield"(%25) : (f32) -> ()
              }) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              "scf.yield"(%24) : (tensor<64x64xf32>) -> ()
            }) : (index, index, index, tensor<64x64xf32>) -> tensor<64x64xf32>
            %13 = "affine.apply"(%5, %arg10) <{map = #map10}> : (index, index) -> index
            %14 = "memref.reinterpret_cast"(%arg2, %13) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: 512, 1>}> : (memref<*xf32>, index) -> memref<64x64xf32, strided<[512, 1], offset: ?>>
            "bufferization.materialize_in_destination"(%12, %14) <{writable}> : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
            "affine.yield"() : () -> ()
          }) {tmd.mapped_to = "y"} : () -> ()
          "affine.yield"() : () -> ()
        }) {tmd.mapped_to = "x"} : () -> ()
      }) : (index, index) -> ()
    }) : (index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

