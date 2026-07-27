  // =====================================================================================
  // Variant: attention__d0i1_d1i1__f01__d_a_a__BB1__BM64__BN128
  //
  //   Naming breakdown:
  //     d0i1_d1i1 : spatial dim 0 (x) → interconnect idx 1 (horizontal_links),
  //                 spatial dim 1 (y) → interconnect idx 1 (vertical_links)
  //     f01       : function arg ordering — arg0=K, arg1=V, arg2=Q (original order)
  //     d_a_a     : copy strategy per operand — Q=direct(d), K^T=all-broadcast(a), V=all-broadcast(a)
  //                 Q copy uses broadcast:[1,1] (direct, no interconnect).
  //                 K^T copy uses [@horizontal_links, @vertical_links], broadcast:[8,8] (all-broadcast).
  //                 V copy uses [@horizontal_links, @vertical_links], broadcast:[8,8] (all-broadcast).
  //     BB=1, BM=64, BN=128 : tile sizes
  //
  //   Global tensor shapes (after per-core tiling by 8x8 mesh):
  //     K  (arg0): memref<32x128x4096xf32>  — [batch_total, head_dim, N_total]
  //     V  (arg1): memref<32x4096x128xf32>  — [batch_total, N_total, head_dim]
  //     Q  (arg2): memref<32x4096x128xf32>  — [batch_total, M_total, head_dim]
  //     O  (arg3): memref<32x4096x128xf32>  — [batch_total, M_total, head_dim]
  //   where BB=1 (batch tile is 1), BM=64, BN=128.
  //         M_total=4096, N_total=4096, head_dim=128.
  //
  //   Loop structure:
  //     Outer: scf.parallel (%arg4, %arg5) in [0,8) x [0,8) — 8x8 spatial cores
  //     Middle: scf.for %arg6 in [0,32) — 32 iterations over batch items
  //       Each iteration processes one batch item (BB=1).
  //       The 64 spatial cores perfectly tile the M_total=4096 dimension: 64 cores * 64 (BM) = 4096.
  //       Core index %13 = %arg4*8 + %arg5 defines the M-tile index [0, 63].
  //     Inner: scf.for %arg7 in [0,32) — 32 iterations over N-dimension tiles
  //       (N_total / BN = 4096 / 128 = 32 iterations)
  //
  //   Interconnect strategy:
  //     Q copy:    direct (no broadcast), each core loads its own Q tile from DRAM
  //     K^T copy:  all-broadcast via horizontal_links + vertical_links, broadcast [8,8]
  //                → one core loads K^T tile, broadcasts to all 64 cores in mesh
  //     V copy:    all-broadcast via horizontal_links + vertical_links, broadcast [8,8]
  //                → one core loads V tile, broadcasts to all 64 cores in mesh
  //
  //   Semaphore model:
  //     loom.semaphore creates a virtual circular buffer (CB) from a physical buffer (loom.alloc).
  //     Multiple CBs can share the same physical buffer, distinguished by semaphore tokens.
  //     This decouples physical memory allocation from synchronization semantics:
  //       - The physical buffer defines the allocated memory on L1.
  //       - Each semaphore CB defines a separate synchronization domain on that memory.
  //     Example: physical buffer %28 (Q tile) has two CBs:
  //       %29 = semaphore(%28) — used for post-loop output writes (acc / l_i)
  //       %37 = semaphore(%28) — used for Q tile DMA load from DRAM
  //     Both CBs share the same physical memory but have independent sync tokens.
  //
  //   Buffer allocation: 11 physical allocs, each with one or more semaphore CBs.
  //   Fills sunk to consumers by SinkFillOps pass.
  //   Q tile buffer (%28) reused for post-loop output.
  // =====================================================================================
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      // arg0 = K (Key matrix, transposed view), shape [batch_total=32, head_dim=128, N_total=4096]
      // arg1 = V (Value matrix),               shape [batch_total=32, N_total=4096, head_dim=128]
      // arg2 = Q (Query matrix),               shape [batch_total=32, M_total=4096, head_dim=128]
      // arg3 = O (Output matrix),              shape [batch_total=32, M_total=4096, head_dim=128]
      %c16384 = arith.constant 16384 : index     // BN * head_dim = 128 * 128
      %c8192 = arith.constant 8192 : index       // BM * head_dim = 64 * 128 = 8192
      %c524288 = arith.constant 524288 : index   // M_total * head_dim = 4096 * 128 = 524288
      %c32 = arith.constant 32 : index           // batch size = 32, also inner loop bound (4096 / 128 = 32)
      %cst = arith.constant 2.000000e+00 : f32   // base for exp2
      %cst_0 = arith.constant 0.12751743074602467 : f64 // qk_scale = 1/sqrt(head_dim) ≈ 1/sqrt(128)
      %cst_1 = arith.constant 0.000000e+00 : f32 // zero (for fill)
      %cst_2 = arith.constant 1.000000e+00 : f32 // one (for l_i init)
      %cst_3 = arith.constant 0xFF800000 : f32   // -inf (for m_i init and amax init)
      %c128 = arith.constant 128 : index         // BN = 128
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index             // spatial mesh dimension (8x8)
      %c1 = arith.constant 1 : index
      // =================== Spatial Parallel: 8x8 core mesh ===================
      // %arg4 = x-dim core index [0,8), %arg5 = y-dim core index [0,8)
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        // =================== Middle Loop: iterate over batch items ===================
        // %arg6 ∈ [0, 32): each core processes all 32 batch items
        scf.for %arg6 = %c0 to %c32 step %c1 {
          // ---- Compute linearized core index (acts as M-tile index) ----
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index                  // (core_x contributes row offset)
          // Since M_total / BM = 4096 / 64 = 64, the 64 cores perfectly map to the 64 M-tiles.
          %13 = arith.addi %12, %arg5 : index          // (linearized core index within 8x8 mesh, range [0, 63])
          // =================== L1 Memory Allocations (11 physical buffers) ===================
          // Each loom.alloc creates a physical buffer on L1.
          // Each loom.semaphore creates a virtual circular buffer (CB) view on a physical buffer.
          // Multiple CBs on the same physical buffer share memory but have independent sync tokens.
          //
          // --- Physical buffer for V tile [BB, BN, head_dim] ---
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>    // V tile phys buf (loaded from DRAM each inner iter)
          //   CB: %43 = semaphore(%14) created inside inner loop for V DMA + matmul use
          //
          // --- Physical buffer for qk/p [BB, BM, BN] ---
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>      // qk = Q @ K^T phys buf, later reused in-place for p
          %16 = loom.semaphore %15 : memref<1x64x128xf32> -> memref<1x64x128xf32>  // CB: qk/p — used for zero-init, matmul output, and softmax computations
          //
          // --- Physical buffer for K^T tile [BB, head_dim, BN] ---
          %17 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>    // K^T tile phys buf (loaded from DRAM each inner iter)
          //   CB: %40 = semaphore(%17) created inside inner loop for K^T DMA + matmul use
          //
          // --- Physical buffer for l_i (running softmax denominator) [BB, BM] ---
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>               // l_i phys buf
          %19 = loom.semaphore %18 : memref<1x64xf32> -> memref<1x64xf32>  // CB: l_i — in-place update across iterations
          //
          // --- Physical buffer for m_i (running row-wise max) [BB, BM] ---
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>               // m_i phys buf
          %21 = loom.semaphore %20 : memref<1x64xf32> -> memref<1x64xf32>  // CB: m_i — updated via copy at end of iter
          //
          // --- Physical buffer for m_ij (new row-wise max) [BB, BM] ---
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>               // m_ij phys buf (also holds intermediate amax)
          %23 = loom.semaphore %22 : memref<1x64xf32> -> memref<1x64xf32>  // CB: m_ij/amax — init with -inf, then overwritten
          //
          // --- Physical buffer for l_ij (row-wise sum of p) [BB, BM] ---
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>               // l_ij phys buf
          %25 = loom.semaphore %24 : memref<1x64xf32> -> memref<1x64xf32>  // CB: l_ij — zeroed then reduced
          //
          // --- Physical buffer for alpha (rescaling factor) [BB, BM] ---
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>               // alpha phys buf
          %27 = loom.semaphore %26 : memref<1x64xf32> -> memref<1x64xf32>  // CB: alpha = 2^(m_i - m_ij)
          //
          // --- Physical buffer for Q tile / output reuse [BB, BM, head_dim] ---
          %28 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>      // Q tile phys buf; REUSED post-loop for output = acc / l_i
          %29 = loom.semaphore %28 : memref<1x64x128xf32> -> memref<1x64x128xf32>  // CB #1: output — used post-loop for acc/l_i result and DRAM store
          //   CB #2: %37 = semaphore(%28) created below for Q tile DMA load
          //   Both CBs share physical buffer %28 but have separate sync domains.
          //
          // --- Physical buffer for acc (accumulator) [BB, BM, head_dim] ---
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>      // acc phys buf (updated in-place across iters)
          %31 = loom.semaphore %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>  // CB: acc — zero-init then rescaled each iter
          //
          // --- Physical buffer for pv (p @ V intermediate) [BB, BM, head_dim] ---
          %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>      // pv scratch phys buf (per-iter)
          %33 = loom.semaphore %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>  // CB: pv — zeroed then used for p@V result
          // =================== Load Q tile from DRAM to L1 ===================
          // Q_tile = Q[batch_idx:batch_idx+1, m_start:m_end, :]
          // where batch_idx = %arg6, m_start = %13 * BM
          //
          // Address calculation for Q (arg2, shape [32, 4096, 128]):
          //   %34 = %arg6 * 524288              (batch offset = batch_idx * M_total * head_dim)
          //   %35 = %13 * 8192                  (M-tile offset = m_tile_idx * BM * head_dim = %13 * 64 * 128)
          //   %36 = %34 + %35                   (Total offset for Q tile)
          //   The view: offset=%36, sizes=[1, 64, 128], strides=[524288, 128, 1]
          %34 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %35 = arith.muli %13, %c8192 : index
          %36 = arith.addi %34, %35 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          // Create CB #2 on physical buffer %28 for Q tile DMA load
          // This is a DIFFERENT semaphore CB than %29 (output), but shares the SAME physical memory.
          // %37 is used as the DMA destination and as the Q operand in Q @ K^T matmul.
          %37 = loom.semaphore %28 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          // DMA: DRAM → L1, load Q tile into CB %37 on phys buf %28 (direct, no broadcast)
          loom.copy %reinterpret_cast, %37 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          // =================== Loop Initialization ===================
          // acc = zeros([BB, BM, head_dim])
          // Memory: writes to CB %31 (on phys %30), out-of-place (fresh init)
          linalg.fill ins(%cst_1 : f32) outs(%31 : memref<1x64x128xf32>)
          // l_i = full([BB, BM], 1.0)
          // Memory: writes to CB %19 (on phys %18), out-of-place (fresh init)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
          // m_i = full([BB, BM], -inf)
          // Memory: writes to CB %21 (on phys %20), out-of-place (fresh init)
          linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
          // =================== Inner Loop: iterate over K/V tiles (32 iters) ===================
          // for n_iter in range(N_total / BN):  (4096 / 128 = 32 iterations)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            // ---- Load K^T tile from DRAM ----
            // K_tile = K[batch_idx:batch_idx+1, :, n_start:n_end]  (transposed view for matmul)
            // where batch_idx = %arg6, n_start = %arg7 * BN
            //
            // Address calculation for K^T (arg0, shape [32, 128, 4096]):
            //   %38 = %arg7 * 128                 (N-tile offset = %arg7 * BN)
            //   %39 = %34 + %38                   (Total offset = batch_offset + N-tile offset)
            //   The view: offset=%39, sizes=[1, 128, 128], strides=[524288, 4096, 1]
            %38 = arith.muli %arg7, %c128 : index
            %39 = arith.addi %34, %38 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            // Create a semaphore CB on physical buffer %17 for this iteration's K^T DMA load
            %40 = loom.semaphore %17 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            // DMA: DRAM → L1, load K^T tile into CB %40 on phys buf %17
            // Interconnect: ALL-BROADCAST via horizontal_links + vertical_links, broadcast [8,8]
            // One core loads and the data is replicated to all 8*8=64 cores.
            loom.copy %reinterpret_cast_5, %40 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            // ---- qk = Q @ K^T ----
            // Zero-init qk buffer, then compute batch matmul
            // Memory: CB %16 (on phys %15) is zeroed then written by batch_matmul (out-of-place, accumulation pattern)
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_1 : f32) outs(%16 : memref<1x64x128xf32>)
            // qk = Q_tile @ K_tile^T, result in CB %16
            // Q from CB %37 (on phys %28), K^T from CB %40 (on phys %17)
            // shapes: [1, 64, 128] @ [1, 128, 128] → [1, 64, 128]
            linalg.batch_matmul ins(%37, %40 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%16 : memref<1x64x128xf32>)
            // ---- amax(qk, -1): row-wise max of qk ----
            // Memory: CB %23 (on phys %22) is first filled with -inf, then reduced over (out-of-place init + in-place reduction)
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%16 : memref<1x64x128xf32>) outs(%23 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            // After this: CB %23 = amax(qk, dim=-1), shape [1, 64]

            // ---- m_ij = max(m_i, amax(qk) * qk_scale) ----
            // Reads m_i from CB %21, reads amax from CB %23, writes result back to CB %23
            // Memory: IN-PLACE on CB %23 — the amax result is overwritten with m_ij.
            //         This is safe because amax is only needed to compute m_ij itself.
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %23 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            // After this: CB %23 = m_ij = max(m_i, amax(qk) * qk_scale)

            // ---- p = exp2(qk * qk_scale - m_ij) ----
            // Reads qk from CB %16, reads m_ij from CB %23, writes p back to CB %16
            // Memory: IN-PLACE on CB %16 — qk is overwritten with p.
            //         qk is no longer needed after this point (consumed by amax above).
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%16 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            // After this: CB %16 = p (attention weights), shape [1, 64, 128]

            // ---- l_ij = sum(p, dim=-1) ----
            // Memory: CB %25 (on phys %24) is zeroed then reduced in-place
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%16 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            // After this: CB %25 = l_ij = sum(p, dim=-1), shape [1, 64]

            // ---- alpha = 2^(m_i - m_ij) ----
            // Reads m_i from CB %21, reads m_ij from CB %23, writes alpha to CB %27
            // Memory: OUT-OF-PLACE, result in separate CB %27 (on phys %26)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %23 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            // After this: CB %27 = alpha = 2^(m_i - m_ij), shape [1, 64]

            // ---- l_i = alpha * l_i + l_ij ----
            // Reads l_i from CB %19, alpha from CB %27, l_ij from CB %25, writes back to CB %19
            // Memory: IN-PLACE on CB %19 (on phys %18) — the running l_i is updated in its own buffer.
            //         This is the key in-place update for the online softmax denominator.
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %27, %25 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            // After this: CB %19 = updated l_i

            // ---- Load V tile from DRAM ----
            // V_tile = V[batch_idx:batch_idx+1, n_start:n_end, :]
            // where batch_idx = %arg6, n_start = %arg7 * BN
            //
            // Address calculation for V (arg1, shape [32, 4096, 128]):
            //   %41 = %arg7 * 16384                 (N-tile offset = %arg7 * BN * head_dim = %arg7 * 128 * 128)
            //   %42 = %34 + %41                   (Total offset = batch_offset + n_tile_offset)
            //   The view: offset=%42, sizes=[1, 128, 128], strides=[524288, 128, 1]
            %41 = arith.muli %arg7, %c16384 : index
            %42 = arith.addi %34, %41 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            // Create a semaphore CB on physical buffer %14 for this iteration's V DMA load
            %43 = loom.semaphore %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            // DMA: DRAM → L1, load V tile into CB %43 on phys buf %14
            // Interconnect: ALL-BROADCAST via horizontal_links + vertical_links, broadcast [8,8]
            // One core loads and the data is replicated to all 8*8=64 cores.
            loom.copy %reinterpret_cast_6, %43 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            // ---- pv = p @ V ----
            // Memory: CB %33 (on phys %32) is zeroed then used as accumulator for batch_matmul (out-of-place)
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_1 : f32) outs(%33 : memref<1x64x128xf32>)
            // p @ V: [1, 64, 128] @ [1, 128, 128] → [1, 64, 128], result in CB %33
            // p from CB %16 (on phys %15), V from CB %43 (on phys %14)
            linalg.batch_matmul ins(%16, %43 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%33 : memref<1x64x128xf32>)
            // ---- acc = pv + acc * alpha ----
            // Reads pv from CB %33, reads previous acc from CB %31, reads alpha from CB %27, writes to CB %31
            // Memory: IN-PLACE on CB %31 (on phys %30) — the running accumulator is updated.
            //         This is the core FlashAttention rescaling: acc = (p @ V) + acc * alpha
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %31, %27 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            // After this: CB %31 = updated acc

            // ---- m_i = m_ij (carry forward the max for next iteration) ----
            // Memory: explicit copy from CB %23 (m_ij) to CB %21 (m_i).
            // This is needed because OSB cannot alias m_i and m_ij (they are live simultaneously
            // during the alpha computation), so a separate copy is required at iteration end.
            linalg.copy ins(%23 : memref<1x64xf32>) outs(%21 : memref<1x64xf32>)
          }
          // =================== Post-loop: normalize by l_i ===================
          // output = acc / l_i
          // Reads final acc from CB %31, final l_i from CB %19, writes to CB %29
          // Memory: OUT-OF-PLACE — result goes to CB %29 (on phys %28, which previously held Q tile via CB %37).
          //         Q is no longer needed after the inner loop, so phys %28 is safely reused here.
          //         CB %29 vs CB %37: different semaphore tokens on same phys buf, ensuring no sync conflict.
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%29 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %38 = arith.divf %in, %in_5 : f32
            linalg.yield %38 : f32
          }
          // =================== Store output tile to DRAM ===================
          // O[batch_idx:batch_idx+1, m_start:m_end, :] = output
          //
          // Address calculation for O (arg3, shape [32, 4096, 128]):
          //   Reuses %36 = batch_offset + m_tile_offset (same offset as Q load, since Q and O share the same tiling)
          //   The view: offset=%36, sizes=[1, 64, 128], strides=[524288, 128, 1]
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%36], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          // DMA: L1 → DRAM, store result from CB %29 to output (direct, no broadcast)
          loom.copy %29, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }