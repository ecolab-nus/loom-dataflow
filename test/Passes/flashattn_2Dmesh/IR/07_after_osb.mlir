module {
  module attributes {loom.pass_name = "Materialize"} {
    // =====================================================================================
    // Variant 1: BB=64, BM=64, BN=64
    //   Symbolic shapes resolved: BB->64, BM->64, BN->64
    //   Outer parallel loop: linearized to 64 iterations (ceildiv(2,64)*ceildiv(4096,64) = 1*64)
    //   Inner loop: 64 iterations over N-dimension (ceildiv(4096,64)=64)
    //
    //   Compared to the previous version, 2 redundant allocs have been removed:
    //   - The old unused %7 alloc (phantom VB from memory binding) is gone.
    //   - The old separate output-division buffer %3 is gone; %8 (Q tile) is now reused
    //     for the post-loop division result since Q is no longer live after the inner loop.
    //   Additionally, linalg.fill ops have been sunk closer to their consumers by the
    //   SinkFillOps pass, reducing buffer liveness and enabling the above reuse.
    // =====================================================================================
    func.func @attention__BB64__BM64__BN64(%arg0: memref<2x4096x4096xf32>, %arg1: memref<2x4096x4096xf32>, %arg2: memref<2x4096x4096xf32>, %arg3: memref<2x4096x4096xf32>) {
      // arg0 = K (Key matrix),    shape [2, 4096, 4096]
      // arg1 = V (Value matrix),  shape [2, 4096, 4096]
      // arg2 = Q (Query matrix),  shape [2, 4096, 4096]
      // arg3 = O (Output matrix), shape [2, 4096, 4096]
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.022542110000000001 : f64 // qk_scale
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32 // -inf
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // NOTE: The original two-level affine.parallel (B, M) with ranges (ceildiv(2,64), ceildiv(4096,64))
      // has been linearized into a single scf.parallel with 64 iterations.
      // Since ceildiv(2,64)=1, the effective iteration count = 1 * 64 = 64.
      // %arg4 is the linearized tile index encoding both batch and M-tile.
      scf.parallel (%arg4) = (%c0) to (%c64) step (%c1) {
        // =================== L1 Memory Allocations ===================
        // The memory binding pass + graph coloring assigned 11 L1 buffers (down from 13).
        // The SinkFillOps pass sunk fill operations closer to consumers, which shortened
        // buffer liveness and enabled graph coloring to reuse %8 for both Q tile and
        // post-loop output.
        //
        // --- 3D buffers ---
        %0 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>   // V tile (loaded from DRAM each inner iter)
        %1 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf32>       // qk = Q @ K^T, later reused in-place for p
        %2 = loom.alloc [64, 4096, 64] on @L1 : memref<64x4096x64xf32>   // K^T tile (loaded from DRAM each inner iter)
        // --- 2D buffers (64x64) ---
        %3 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>              // l_i: running softmax denominator (in-place across iters)
        %4 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>              // m_i: running row-wise max (updated via copy at end of iter)
        %5 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>              // m_ij: new row-wise max for current iter (also holds intermediate amax)
        %6 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>              // l_ij: row-wise sum of p for current iter
        %7 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>              // alpha = 2^(m_i - m_ij), rescaling factor
        // --- 3D buffers (reused / accumulator) ---
        %8 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>   // Q tile (loaded once); REUSED post-loop for output = acc / l_i
        %9 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>   // acc (accumulator, updated in-place across iters)
        %10 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>  // p @ V intermediate result (per-iter scratch)

        // =================== Load Q tile from DRAM to L1 ===================
        // Q_tile = Q[batch, m_start:m_end, :]
        // Computed once before the inner loop (Q is loop-invariant).
        // %11 = iter_B * BB * BM * head_dim (linearized offset into Q)
        %11 = arith.muli %arg4, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        // DMA: DRAM -> L1, load Q tile into %8
        loom.copy %reinterpret_cast, %8 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x64x4096xf32>

        // =================== Loop Initialization ===================
        // acc = zeros([BB, BM, head_dim])
        // Memory: writes to %9, out-of-place (fresh init)
        linalg.fill ins(%cst_1 : f32) outs(%9 : memref<64x64x4096xf32>)
        // l_i = full([BB, BM], 1.0)
        // Memory: writes to %3, out-of-place (fresh init)
        linalg.fill ins(%cst_2 : f32) outs(%3 : memref<64x64xf32>)
        // m_i = full([BB, BM], -inf)
        // Memory: writes to %4, out-of-place (fresh init)
        linalg.fill ins(%cst_3 : f32) outs(%4 : memref<64x64xf32>)

        // =================== Inner Loop: iterate over K/V tiles ===================
        // for n_iter in range(ceildiv(4096, BN)):  (64 iterations)
        scf.for %arg5 = %c0 to %c64 step %c1 {
          // ---- Load K^T tile from DRAM ----
          // K_tile = K[batch, :, n_start:n_end]  (transposed view for matmul)
          %12 = arith.muli %arg5, %c64 : index  // iter_N * BN
          %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [64, 4096, 64], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x4096x64xf32, strided<[16777216, 4096, 1], offset: ?>>
          // DMA: DRAM -> L1, load K^T tile into %2
          loom.copy %reinterpret_cast_5, %2 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x4096x64xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x4096x64xf32>

          // ---- qk = Q @ K^T ----
          // Zero-init qk buffer, then compute batch matmul
          // Memory: %1 is zeroed then written by batch_matmul (out-of-place, accumulation pattern)
          // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
          linalg.fill ins(%cst_1 : f32) outs(%1 : memref<64x64x64xf32>)
          // qk = Q_tile @ K_tile^T, result in %1
          linalg.batch_matmul ins(%8, %2 : memref<64x64x4096xf32>, memref<64x4096x64xf32>) outs(%1 : memref<64x64x64xf32>)

          // ---- amax(qk, -1): row-wise max of qk ----
          // Memory: %5 is first filled with -inf, then reduced over (out-of-place init + in-place reduction)
          // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
          linalg.fill ins(%cst_3 : f32) outs(%5 : memref<64x64xf32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : memref<64x64x64xf32>) outs(%5 : memref<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %14 = arith.maximumf %in, %out : f32
            linalg.yield %14 : f32
          }
          // After this: %5 = amax(qk, dim=-1)

          // ---- m_ij = max(m_i, amax(qk) * qk_scale) ----
          // Reads m_i from %4, reads amax from %5, writes result back to %5
          // Memory: IN-PLACE on %5 — the amax result is overwritten with m_ij.
          //         This is safe because amax is only needed to compute m_ij itself.
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : memref<64x64xf32>, memref<64x64xf32>) outs(%5 : memref<64x64xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %14 = arith.truncf %cst_0 : f64 to f32
            %15 = arith.mulf %in_7, %14 : f32
            %16 = arith.cmpf ogt, %in, %15 : f32
            %17 = arith.select %16, %in, %15 : f32
            linalg.yield %17 : f32
          }
          // After this: %5 = m_ij

          // ---- p = exp2(qk * qk_scale - m_ij) ----
          // Reads qk from %1, reads m_ij from %5, writes p back to %1
          // Memory: IN-PLACE on %1 — qk is overwritten with p.
          //         qk is no longer needed after this point (it was consumed by amax above).
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %5 : memref<64x64x64xf32>, memref<64x64xf32>) outs(%1 : memref<64x64x64xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %14 = arith.truncf %cst_0 : f64 to f32
            %15 = arith.mulf %in, %14 : f32
            %16 = arith.subf %15, %in_7 : f32
            %17 = math.powf %cst, %16 : f32
            linalg.yield %17 : f32
          }
          // After this: %1 = p (attention weights)

          // ---- l_ij = sum(p, dim=-1) ----
          // Memory: %6 is zeroed then reduced in-place
          // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
          linalg.fill ins(%cst_1 : f32) outs(%6 : memref<64x64xf32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : memref<64x64x64xf32>) outs(%6 : memref<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %14 = arith.addf %in, %out : f32
            linalg.yield %14 : f32
          }
          // After this: %6 = l_ij

          // ---- alpha = 2^(m_i - m_ij) ----
          // Reads m_i from %4, reads m_ij from %5, writes alpha to %7
          // Memory: OUT-OF-PLACE, result in separate buffer %7
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : memref<64x64xf32>, memref<64x64xf32>) outs(%7 : memref<64x64xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %14 = arith.subf %in, %in_7 : f32
            %15 = math.powf %cst, %14 : f32
            linalg.yield %15 : f32
          }
          // After this: %7 = alpha

          // ---- l_i = alpha * l_i + l_ij ----
          // Reads l_i from %3, alpha from %7, l_ij from %6, writes back to %3
          // Memory: IN-PLACE on %3 — the running l_i is updated in its own buffer.
          //         This is the key in-place update for the online softmax denominator.
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %7, %6 : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>) outs(%3 : memref<64x64xf32>) {
          ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
            %14 = arith.mulf %in, %in_7 : f32
            %15 = arith.addf %14, %in_8 : f32
            linalg.yield %15 : f32
          }
          // After this: %3 = updated l_i

          // ---- Load V tile from DRAM ----
          // V_tile = V[batch, n_start:n_end, :]
          %13 = arith.muli %arg5, %c262144 : index  // iter_N * BN * head_dim (linearized offset into V)
          %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          // DMA: DRAM -> L1, load V tile into %0
          loom.copy %reinterpret_cast_6, %0 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x64x4096xf32>

          // ---- pv = p @ V ----
          // Memory: %10 is zeroed then used as accumulator for batch_matmul (out-of-place)
          // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
          linalg.fill ins(%cst_1 : f32) outs(%10 : memref<64x64x4096xf32>)
          // p @ V, result in %10
          linalg.batch_matmul ins(%1, %0 : memref<64x64x64xf32>, memref<64x64x4096xf32>) outs(%10 : memref<64x64x4096xf32>)

          // ---- acc = pv + acc * alpha ----
          // Reads pv from %10, reads previous acc from %9, reads alpha from %7, writes to %9
          // Memory: IN-PLACE on %9 — the running accumulator is updated.
          //         This is the core FlashAttention rescaling: acc = (p @ V) + acc * alpha
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %9, %7 : memref<64x64x4096xf32>, memref<64x64x4096xf32>, memref<64x64xf32>) outs(%9 : memref<64x64x4096xf32>) {
          ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
            %14 = arith.mulf %in_7, %in_8 : f32
            %15 = arith.addf %in, %14 : f32
            linalg.yield %15 : f32
          }
          // After this: %9 = updated acc

          // ---- m_i = m_ij (carry forward the max for next iteration) ----
          // Memory: explicit copy from %5 (m_ij) to %4 (m_i).
          // This is needed because OSB cannot alias m_i and m_ij (they are live simultaneously
          // during the alpha computation), so a separate copy is required at iteration end.
          linalg.copy ins(%5 : memref<64x64xf32>) outs(%4 : memref<64x64xf32>)
        }
        // =================== Post-loop: normalize by l_i ===================
        // output = acc / l_i
        // Reads final acc from %9, final l_i from %3, writes to %8
        // Memory: OUT-OF-PLACE — result goes to %8 (which previously held Q tile).
        //         Q is no longer needed after the inner loop, so %8 is safely reused here.
        //         This is an improvement: the old version needed a separate buffer %3 for this.
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %3 : memref<64x64x4096xf32>, memref<64x64xf32>) outs(%8 : memref<64x64x4096xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %12 = arith.divf %in, %in_5 : f32
          linalg.yield %12 : f32
        }

        // =================== Store output tile to DRAM ===================
        // O[batch, m_start:m_end, :] = output
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%11], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        // DMA: L1 -> DRAM, store result from %8 to output
        loom.copy %8, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64x4096xf32>, memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        scf.reduce 
      }
      return
    }

    // =====================================================================================
    // Variant 2: BB=32, BM=32, BN=256
    //   Symbolic shapes resolved: BB->32, BM->32, BN->256
    //   Outer parallel loop: linearized to 128 iterations (ceildiv(2,32)*ceildiv(4096,32) = 1*128)
    //   Inner loop: 16 iterations over N-dimension (ceildiv(4096,256)=16)
    //
    //   Same structural improvements as Variant 1: 11 allocs (down from 13),
    //   fills sunk to consumers, Q tile buffer reused for post-loop output.
    // =====================================================================================
    func.func @attention__BB32__BM32__BN256(%arg0: memref<2x4096x4096xf32>, %arg1: memref<2x4096x4096xf32>, %arg2: memref<2x4096x4096xf32>, %arg3: memref<2x4096x4096xf32>) {
      // arg0 = K (Key matrix),    shape [2, 4096, 4096]
      // arg1 = V (Value matrix),  shape [2, 4096, 4096]
      // arg2 = Q (Query matrix),  shape [2, 4096, 4096]
      // arg3 = O (Output matrix), shape [2, 4096, 4096]
      %c1048576 = arith.constant 1048576 : index
      %c131072 = arith.constant 131072 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.022542110000000001 : f64 // qk_scale
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32 // -inf
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      // Linearized parallel over batch and M-tiles: 128 iterations
      scf.parallel (%arg4) = (%c0) to (%c128) step (%c1) {
        // =================== L1 Memory Allocations ===================
        // Same structure as Variant 1 but with different tile sizes.
        // 11 L1 buffers total (down from 13 in old version).
        //
        // --- 3D buffers ---
        %0 = loom.alloc [32, 256, 4096] on @L1 : memref<32x256x4096xf32>  // V tile (loaded from DRAM each inner iter)
        %1 = loom.alloc [32, 32, 256] on @L1 : memref<32x32x256xf32>      // qk = Q @ K^T, later reused in-place for p
        %2 = loom.alloc [32, 4096, 256] on @L1 : memref<32x4096x256xf32>  // K^T tile (loaded from DRAM each inner iter)
        // --- 2D buffers (32x32) ---
        %3 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>               // l_i: running softmax denominator (in-place across iters)
        %4 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>               // m_i: running row-wise max (updated via copy at end of iter)
        %5 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>               // m_ij: new row-wise max (also holds intermediate amax)
        %6 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>               // l_ij: row-wise sum of p for current iter
        %7 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>               // alpha = 2^(m_i - m_ij), rescaling factor
        // --- 3D buffers (reused / accumulator) ---
        %8 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>    // Q tile (loaded once); REUSED post-loop for output = acc / l_i
        %9 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>    // acc (accumulator, updated in-place across iters)
        %10 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>   // p @ V intermediate result (per-iter scratch)

        // =================== Load Q tile from DRAM to L1 ===================
        // %11 = linearized offset for Q tile
        %11 = arith.muli %arg4, %c131072 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        // DMA: DRAM -> L1, load Q tile into %8
        loom.copy %reinterpret_cast, %8 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x32x4096xf32>

        // =================== Loop Initialization ===================
        // acc = zeros([BB, BM, head_dim])
        linalg.fill ins(%cst_1 : f32) outs(%9 : memref<32x32x4096xf32>)
        // l_i = full([BB, BM], 1.0)
        linalg.fill ins(%cst_2 : f32) outs(%3 : memref<32x32xf32>)
        // m_i = full([BB, BM], -inf)
        linalg.fill ins(%cst_3 : f32) outs(%4 : memref<32x32xf32>)

        // =================== Inner Loop: iterate over K/V tiles (16 iters) ===================
        scf.for %arg5 = %c0 to %c16 step %c1 {
          // ---- Load K^T tile from DRAM ----
          %12 = arith.muli %arg5, %c256 : index  // iter_N * BN
          %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 4096, 256], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x4096x256xf32, strided<[16777216, 4096, 1], offset: ?>>
          // DMA: DRAM -> L1, load K^T tile into %2
          loom.copy %reinterpret_cast_5, %2 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x4096x256xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x4096x256xf32>

          // ---- qk = Q @ K^T ----
          // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
          linalg.fill ins(%cst_1 : f32) outs(%1 : memref<32x32x256xf32>)
          linalg.batch_matmul ins(%8, %2 : memref<32x32x4096xf32>, memref<32x4096x256xf32>) outs(%1 : memref<32x32x256xf32>)

          // ---- amax(qk, -1) ----
          // Memory: %5 filled with -inf, then reduced in-place
          // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
          linalg.fill ins(%cst_3 : f32) outs(%5 : memref<32x32xf32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : memref<32x32x256xf32>) outs(%5 : memref<32x32xf32>) {
          ^bb0(%in: f32, %out: f32):
            %14 = arith.maximumf %in, %out : f32
            linalg.yield %14 : f32
          }

          // ---- m_ij = max(m_i, amax(qk) * qk_scale) ----
          // Memory: IN-PLACE on %5 (amax overwritten with m_ij)
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : memref<32x32xf32>, memref<32x32xf32>) outs(%5 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %14 = arith.truncf %cst_0 : f64 to f32
            %15 = arith.mulf %in_7, %14 : f32
            %16 = arith.cmpf ogt, %in, %15 : f32
            %17 = arith.select %16, %in, %15 : f32
            linalg.yield %17 : f32
          }

          // ---- p = exp2(qk * qk_scale - m_ij) ----
          // Memory: IN-PLACE on %1 (qk overwritten with p)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %5 : memref<32x32x256xf32>, memref<32x32xf32>) outs(%1 : memref<32x32x256xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %14 = arith.truncf %cst_0 : f64 to f32
            %15 = arith.mulf %in, %14 : f32
            %16 = arith.subf %15, %in_7 : f32
            %17 = math.powf %cst, %16 : f32
            linalg.yield %17 : f32
          }

          // ---- l_ij = sum(p, dim=-1) ----
          // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
          linalg.fill ins(%cst_1 : f32) outs(%6 : memref<32x32xf32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : memref<32x32x256xf32>) outs(%6 : memref<32x32xf32>) {
          ^bb0(%in: f32, %out: f32):
            %14 = arith.addf %in, %out : f32
            linalg.yield %14 : f32
          }

          // ---- alpha = 2^(m_i - m_ij) ----
          // Memory: OUT-OF-PLACE, result in %7
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : memref<32x32xf32>, memref<32x32xf32>) outs(%7 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %14 = arith.subf %in, %in_7 : f32
            %15 = math.powf %cst, %14 : f32
            linalg.yield %15 : f32
          }

          // ---- l_i = alpha * l_i + l_ij ----
          // Memory: IN-PLACE on %3
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %7, %6 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%3 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
            %14 = arith.mulf %in, %in_7 : f32
            %15 = arith.addf %14, %in_8 : f32
            linalg.yield %15 : f32
          }

          // ---- Load V tile from DRAM ----
          %13 = arith.muli %arg5, %c1048576 : index  // iter_N * BN * head_dim
          %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 256, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x256x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          // DMA: DRAM -> L1, load V tile into %0
          loom.copy %reinterpret_cast_6, %0 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x256x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x256x4096xf32>

          // ---- pv = p @ V ----
          // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
          linalg.fill ins(%cst_1 : f32) outs(%10 : memref<32x32x4096xf32>)
          linalg.batch_matmul ins(%1, %0 : memref<32x32x256xf32>, memref<32x256x4096xf32>) outs(%10 : memref<32x32x4096xf32>)

          // ---- acc = pv + acc * alpha ----
          // Memory: IN-PLACE on %9
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %9, %7 : memref<32x32x4096xf32>, memref<32x32x4096xf32>, memref<32x32xf32>) outs(%9 : memref<32x32x4096xf32>) {
          ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
            %14 = arith.mulf %in_7, %in_8 : f32
            %15 = arith.addf %in, %14 : f32
            linalg.yield %15 : f32
          }

          // ---- m_i = m_ij ----
          // Memory: explicit copy %5 -> %4
          linalg.copy ins(%5 : memref<32x32xf32>) outs(%4 : memref<32x32xf32>)
        }
        // =================== Post-loop: output = acc / l_i ===================
        // Memory: OUT-OF-PLACE, reads %9 (acc), %3 (l_i), writes to %8 (reused Q tile buffer)
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %3 : memref<32x32x4096xf32>, memref<32x32xf32>) outs(%8 : memref<32x32x4096xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %12 = arith.divf %in, %in_5 : f32
          linalg.yield %12 : f32
        }

        // =================== Store output tile to DRAM ===================
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%11], sizes: [32, 32, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        // DMA: L1 -> DRAM, store result from %8 to output
        loom.copy %8, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x4096xf32>, memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        scf.reduce 
      }
      return
    }
  }
}
