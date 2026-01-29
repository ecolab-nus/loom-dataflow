// matmul_kernel__d0i0_d1i1__f01__c0mem_c1mem__compute
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "firmware_common.h"
#include "llk_defs.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/activations.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_unary/rounding.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#include "compute_kernel_api/eltwise_unary/gelu.h"
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/eltwise_unary/typecast.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/eltwise_unary/bitwise_not.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/where.h"
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL
#include "compute_kernel_api/reduce.h"

// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MATMUL_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MATMUL_LLKS_H

namespace experimental {

ALWI void matmul_block(uint32_t in0_cb_id, uint32_t in1_cb_id,
                       uint32_t in0_tile_index, uint32_t in1_tile_index,
                       uint32_t idst, const uint32_t transpose, uint32_t ct_dim,
                       uint32_t rt_dim, uint32_t kt_dim, uint32_t nt_dim) {

  for (uint32_t i = 0; i < kt_dim; i++) {
    ckernel::matmul_block(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index,
                          idst, transpose, ct_dim, rt_dim, kt_dim);
    in0_tile_index++;
    in1_tile_index += nt_dim;
  }
}

} // namespace experimental

#endif

namespace NAMESPACE {
void kernel_main() {
  int32_t v1 = 10;
  int32_t v2 = 9;
  size_t v3 = 0;
  size_t v4 = 8;
  size_t v5 = 1;
  int32_t v6 = 8;
  int32_t v7 = 7;
  int32_t v8 = 6;
  int32_t v9 = 5;
  int32_t v10 = 4;
  int32_t v11 = 3;
  int32_t v12 = 2;
  int32_t v13 = 1;
  int32_t v14 = 0;
  ::tt::CB v15 = get_arg_val<uint32_t>(v14);
  int32_t v16 = get_arg_val<uint32_t>(v13);
  ::tt::CB v17 = get_arg_val<uint32_t>(v12);
  int32_t v18 = get_arg_val<uint32_t>(v11);
  ::tt::CB v19 = get_arg_val<uint32_t>(v10);
  int32_t v20 = get_arg_val<uint32_t>(v9);
  int32_t v21 = get_arg_val<uint32_t>(v8);
  ptrdiff_t v22 = (ptrdiff_t) v21;
  size_t v23 = (size_t) v22;
  int32_t v24 = get_arg_val<uint32_t>(v7);
  ptrdiff_t v25 = (ptrdiff_t) v24;
  size_t v26 = (size_t) v25;
  int32_t v27 = get_arg_val<uint32_t>(v6);
  ::tt::CB v28 = get_arg_val<uint32_t>(v14);
  mm_block_init(v15, v17, v28, v14, v13, v13, v13);
  tile_regs_acquire();
  int32_t v29 = get_arg_val<uint32_t>(v2);
  int32_t v30 = get_arg_val<uint32_t>(v1);
  ptrdiff_t v31 = (ptrdiff_t) v23;
  ptrdiff_t v32 = (ptrdiff_t) v3;
  bool v33 = v31 <= v32;
  size_t v34 = v3 - v23;
  size_t v35 = v23 - v5;
  size_t v36 = v33 ? v34 : v35;
  size_t v37 = v36 / v4;
  size_t v38 = v3 - v37;
  size_t v39 = v37 + v5;
  size_t v40 = v33 ? v38 : v39;
  for (size_t i41 = v3; i41 < v40; i41 += v5) {
    ptrdiff_t v42 = (ptrdiff_t) v26;
    ptrdiff_t v43 = (ptrdiff_t) v3;
    bool v44 = v42 <= v43;
    size_t v45 = v3 - v26;
    size_t v46 = v26 - v5;
    size_t v47 = v44 ? v45 : v46;
    size_t v48 = v47 / v4;
    size_t v49 = v3 - v48;
    size_t v50 = v48 + v5;
    size_t v51 = v44 ? v49 : v50;
    for (size_t j52 = v3; j52 < v51; j52 += v5) {
      for (size_t k53 = v3; k53 < v4; k53 += v5) {
        cb_wait_front(v15, v10);
        cb_wait_front(v17, v10);
        experimental::matmul_block(v15, v17, v14, v14, v14, v14, v13, v13, v13, v13);
        cb_pop_front(v15, v10);
        cb_pop_front(v17, v10);
      }
      cb_reserve_back(v19, v10);
      tile_regs_commit();
      tile_regs_wait();
      for (int32_t k54 = v14; k54 < v10; k54 += v13) {
        pack_tile<false>(k54, v19, k54);
      }
      tile_regs_release();
      cb_push_back(v19, v10);
    }
  }
  return;
}
void MAIN { kernel_main(); }
}

// matmul_kernel__d0i0_d1i1__f01__c0mem_c1mem__reader
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "firmware_common.h"
#include "dataflow_api.h"
void kernel_main() {
  size_t v1 = 512;
  size_t v2 = 32768;
  size_t v3 = 4096;
  size_t v4 = 64;
  int32_t v5 = 16384;
  int32_t v6 = 32;
  int32_t v7 = 8192;
  int32_t v8 = 10;
  int32_t v9 = 9;
  size_t v10 = 0;
  size_t v11 = 8;
  size_t v12 = 1;
  int32_t v13 = 8;
  int32_t v14 = 7;
  int32_t v15 = 6;
  int32_t v16 = 5;
  int32_t v17 = 4;
  int32_t v18 = 3;
  int32_t v19 = 2;
  int32_t v20 = 1;
  int32_t v21 = 0;
  ::tt::CB v22 = get_arg_val<uint32_t>(v21);
  int32_t v23 = get_arg_val<uint32_t>(v20);
  int32_t v24 = get_tile_size(v22);
  int32_t v25 = get_tile_size(v22);
  TensorAccessorArgs v26 = TensorAccessorArgs<0, 0>();
  TensorAccessor v27 = TensorAccessor(v26, v23, v25);
  ::tt::CB v28 = get_arg_val<uint32_t>(v19);
  int32_t v29 = get_arg_val<uint32_t>(v18);
  int32_t v30 = get_tile_size(v28);
  int32_t v31 = get_tile_size(v28);
  TensorAccessorArgs v32 = TensorAccessorArgs<1, 1>();
  TensorAccessor v33 = TensorAccessor(v32, v29, v31);
  ::tt::CB v34 = get_arg_val<uint32_t>(v17);
  int32_t v35 = get_arg_val<uint32_t>(v16);
  int32_t v36 = get_tile_size(v34);
  TensorAccessorArgs v37 = TensorAccessorArgs<2, 2>();
  TensorAccessor v38 = TensorAccessor(v37, v35, v36);
  int32_t v39 = get_arg_val<uint32_t>(v15);
  ptrdiff_t v40 = (ptrdiff_t) v39;
  size_t v41 = (size_t) v40;
  int32_t v42 = get_arg_val<uint32_t>(v14);
  ptrdiff_t v43 = (ptrdiff_t) v42;
  size_t v44 = (size_t) v43;
  int32_t v45 = get_arg_val<uint32_t>(v13);
  int32_t v46 = get_arg_val<uint32_t>(v9);
  ptrdiff_t v47 = (ptrdiff_t) v46;
  size_t v48 = (size_t) v47;
  int32_t v49 = get_arg_val<uint32_t>(v8);
  ptrdiff_t v50 = (ptrdiff_t) v49;
  size_t v51 = (size_t) v50;
  ptrdiff_t v52 = (ptrdiff_t) v41;
  ptrdiff_t v53 = (ptrdiff_t) v10;
  bool v54 = v52 <= v53;
  size_t v55 = v10 - v41;
  size_t v56 = v41 - v12;
  size_t v57 = v54 ? v55 : v56;
  size_t v58 = v57 / v11;
  size_t v59 = v10 - v58;
  size_t v60 = v58 + v12;
  size_t v61 = v54 ? v59 : v60;
  for (size_t i62 = v10; i62 < v61; i62 += v12) {
    ptrdiff_t v63 = (ptrdiff_t) v44;
    ptrdiff_t v64 = (ptrdiff_t) v10;
    bool v65 = v63 <= v64;
    size_t v66 = v10 - v44;
    size_t v67 = v44 - v12;
    size_t v68 = v65 ? v66 : v67;
    size_t v69 = v68 / v11;
    size_t v70 = v10 - v69;
    size_t v71 = v69 + v12;
    size_t v72 = v65 ? v70 : v71;
    for (size_t j73 = v10; j73 < v72; j73 += v12) {
      for (size_t k74 = v10; k74 < v11; k74 += v12) {
        size_t v75 = k74 * v4;
        size_t v76 = i62 * v3;
        size_t v77 = v75 + v76;
        size_t v78 = v48 * v2;
        size_t v79 = v77 + v78;
        ptrdiff_t v80 = (ptrdiff_t) v79;
        int32_t v81 = (int32_t) v80;
        uint32_t v82 = (uint32_t) v7;
        uint32_t v83 = (uint32_t) v24;
        uint32_t v84 = v82 / v83;
        int32_t v85 = (int32_t) v84;
        cb_reserve_back(v22, v85);
        int32_t v86 = get_write_ptr(v22);
        int32_t v87;
        v87 = v86;
        for (int32_t l88 = v21; l88 < v19; l88 += v20) {
          int32_t v89 = v87;
          int32_t v90;
          v90 = v89;
          for (int32_t m91 = v21; m91 < v19; m91 += v20) {
            int32_t v92 = v90;
            uint32_t v93 = (uint32_t) l88;
            uint32_t v94 = (uint32_t) v5;
            uint32_t v95 = v93 * v94;
            int32_t v96 = (int32_t) v95;
            uint32_t v97 = (uint32_t) m91;
            uint32_t v98 = (uint32_t) v6;
            uint32_t v99 = v97 * v98;
            int32_t v100 = (int32_t) v99;
            uint32_t v101 = (uint32_t) v96;
            uint32_t v102 = (uint32_t) v100;
            uint32_t v103 = v101 + v102;
            int32_t v104 = (int32_t) v103;
            uint32_t v105 = (uint32_t) v81;
            uint32_t v106 = (uint32_t) v104;
            uint32_t v107 = v105 + v106;
            int32_t v108 = (int32_t) v107;
            uint32_t v109 = (uint32_t) v108;
            uint32_t v110 = (uint32_t) v19;
            uint32_t v111 = v109 * v110;
            int32_t v112 = (int32_t) v111;
            uint32_t v113 = (uint32_t) v112;
            uint32_t v114 = (uint32_t) v24;
            uint32_t v115 = v113 / v114;
            int32_t v116 = (int32_t) v115;
            noc_async_read_tile(v116, v27, v92);
            uint32_t v117 = (uint32_t) v92;
            uint32_t v118 = (uint32_t) v24;
            uint32_t v119 = v117 + v118;
            int32_t v120 = (int32_t) v119;
            v90 = v120;
          }
          int32_t v121 = v90;
          v87 = v121;
        }
        noc_async_read_barrier();
        cb_push_back(v22, v85);
        size_t v122 = k74 * v2;
        size_t v123 = j73 * v4;
        size_t v124 = v122 + v123;
        size_t v125 = v51 * v1;
        size_t v126 = v124 + v125;
        ptrdiff_t v127 = (ptrdiff_t) v126;
        int32_t v128 = (int32_t) v127;
        uint32_t v129 = (uint32_t) v7;
        uint32_t v130 = (uint32_t) v30;
        uint32_t v131 = v129 / v130;
        int32_t v132 = (int32_t) v131;
        cb_reserve_back(v28, v132);
        int32_t v133 = get_write_ptr(v28);
        int32_t v134;
        v134 = v133;
        for (int32_t l135 = v21; l135 < v19; l135 += v20) {
          int32_t v136 = v134;
          int32_t v137;
          v137 = v136;
          for (int32_t m138 = v21; m138 < v19; m138 += v20) {
            int32_t v139 = v137;
            uint32_t v140 = (uint32_t) l135;
            uint32_t v141 = (uint32_t) v5;
            uint32_t v142 = v140 * v141;
            int32_t v143 = (int32_t) v142;
            uint32_t v144 = (uint32_t) m138;
            uint32_t v145 = (uint32_t) v6;
            uint32_t v146 = v144 * v145;
            int32_t v147 = (int32_t) v146;
            uint32_t v148 = (uint32_t) v143;
            uint32_t v149 = (uint32_t) v147;
            uint32_t v150 = v148 + v149;
            int32_t v151 = (int32_t) v150;
            uint32_t v152 = (uint32_t) v128;
            uint32_t v153 = (uint32_t) v151;
            uint32_t v154 = v152 + v153;
            int32_t v155 = (int32_t) v154;
            uint32_t v156 = (uint32_t) v155;
            uint32_t v157 = (uint32_t) v19;
            uint32_t v158 = v156 * v157;
            int32_t v159 = (int32_t) v158;
            uint32_t v160 = (uint32_t) v159;
            uint32_t v161 = (uint32_t) v30;
            uint32_t v162 = v160 / v161;
            int32_t v163 = (int32_t) v162;
            noc_async_read_tile(v163, v33, v139);
            uint32_t v164 = (uint32_t) v139;
            uint32_t v165 = (uint32_t) v30;
            uint32_t v166 = v164 + v165;
            int32_t v167 = (int32_t) v166;
            v137 = v167;
          }
          int32_t v168 = v137;
          v134 = v168;
        }
        noc_async_read_barrier();
        cb_push_back(v28, v132);
      }
    }
  }
  return;
}

// matmul_kernel__d0i0_d1i1__f01__c0mem_c1mem__writer
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "firmware_common.h"
#include "dataflow_api.h"
void kernel_main() {
  size_t v1 = 32768;
  size_t v2 = 64;
  size_t v3 = 4096;
  size_t v4 = 1;
  size_t v5 = 8;
  size_t v6 = 0;
  int32_t v7 = 16;
  int32_t v8 = 16384;
  int32_t v9 = 512;
  int32_t v10 = 32;
  int32_t v11 = 8192;
  int32_t v12 = 10;
  int32_t v13 = 9;
  int32_t v14 = 8;
  int32_t v15 = 7;
  int32_t v16 = 6;
  int32_t v17 = 5;
  int32_t v18 = 4;
  int32_t v19 = 3;
  int32_t v20 = 2;
  int32_t v21 = 1;
  int32_t v22 = 0;
  ::tt::CB v23 = get_arg_val<uint32_t>(v22);
  int32_t v24 = get_arg_val<uint32_t>(v21);
  int32_t v25 = get_tile_size(v23);
  TensorAccessorArgs v26 = TensorAccessorArgs<0, 0>();
  TensorAccessor v27 = TensorAccessor(v26, v24, v25);
  ::tt::CB v28 = get_arg_val<uint32_t>(v20);
  int32_t v29 = get_arg_val<uint32_t>(v19);
  int32_t v30 = get_tile_size(v28);
  TensorAccessorArgs v31 = TensorAccessorArgs<1, 1>();
  TensorAccessor v32 = TensorAccessor(v31, v29, v30);
  ::tt::CB v33 = get_arg_val<uint32_t>(v18);
  int32_t v34 = get_arg_val<uint32_t>(v17);
  int32_t v35 = get_tile_size(v33);
  int32_t v36 = get_tile_size(v33);
  TensorAccessorArgs v37 = TensorAccessorArgs<2, 2>();
  TensorAccessor v38 = TensorAccessor(v37, v34, v36);
  int32_t v39 = get_arg_val<uint32_t>(v16);
  ptrdiff_t v40 = (ptrdiff_t) v39;
  size_t v41 = (size_t) v40;
  int32_t v42 = get_arg_val<uint32_t>(v15);
  ptrdiff_t v43 = (ptrdiff_t) v42;
  size_t v44 = (size_t) v43;
  int32_t v45 = get_arg_val<uint32_t>(v14);
  ::tt::CB v46 = get_arg_val<uint32_t>(v22);
  int32_t v47 = get_arg_val<uint32_t>(v13);
  ptrdiff_t v48 = (ptrdiff_t) v47;
  size_t v49 = (size_t) v48;
  int32_t v50 = get_arg_val<uint32_t>(v12);
  ptrdiff_t v51 = (ptrdiff_t) v50;
  size_t v52 = (size_t) v51;
  ptrdiff_t v53 = (ptrdiff_t) v41;
  ptrdiff_t v54 = (ptrdiff_t) v6;
  bool v55 = v53 <= v54;
  size_t v56 = v6 - v41;
  size_t v57 = v41 - v4;
  size_t v58 = v55 ? v56 : v57;
  size_t v59 = v58 / v5;
  size_t v60 = v6 - v59;
  size_t v61 = v59 + v4;
  size_t v62 = v55 ? v60 : v61;
  for (size_t i63 = v6; i63 < v62; i63 += v4) {
    ptrdiff_t v64 = (ptrdiff_t) v44;
    ptrdiff_t v65 = (ptrdiff_t) v6;
    bool v66 = v64 <= v65;
    size_t v67 = v6 - v44;
    size_t v68 = v44 - v4;
    size_t v69 = v66 ? v67 : v68;
    size_t v70 = v69 / v5;
    size_t v71 = v6 - v70;
    size_t v72 = v70 + v4;
    size_t v73 = v66 ? v71 : v72;
    for (size_t j74 = v6; j74 < v73; j74 += v4) {
      size_t v75 = j74 * v5;
      size_t v76 = i63 * v3;
      size_t v77 = v75 + v76;
      size_t v78 = v52 * v2;
      size_t v79 = v77 + v78;
      size_t v80 = v49 * v1;
      size_t v81 = v79 + v80;
      ptrdiff_t v82 = (ptrdiff_t) v81;
      int32_t v83 = (int32_t) v82;
      uint32_t v84 = (uint32_t) v11;
      uint32_t v85 = (uint32_t) v35;
      uint32_t v86 = v84 / v85;
      int32_t v87 = (int32_t) v86;
      cb_wait_front(v33, v87);
      int32_t v88 = get_read_ptr(v33);
      int32_t v89;
      v89 = v88;
      for (int32_t k90 = v22; k90 < v20; k90 += v21) {
        int32_t v91 = v89;
        int32_t v92;
        v92 = v91;
        for (int32_t l93 = v22; l93 < v20; l93 += v21) {
          int32_t v94 = v92;
          uint32_t v95 = (uint32_t) k90;
          uint32_t v96 = (uint32_t) v8;
          uint32_t v97 = v95 * v96;
          int32_t v98 = (int32_t) v97;
          uint32_t v99 = (uint32_t) l93;
          uint32_t v100 = (uint32_t) v10;
          uint32_t v101 = v99 * v100;
          int32_t v102 = (int32_t) v101;
          uint32_t v103 = (uint32_t) v98;
          uint32_t v104 = (uint32_t) v102;
          uint32_t v105 = v103 + v104;
          int32_t v106 = (int32_t) v105;
          uint32_t v107 = (uint32_t) v83;
          uint32_t v108 = (uint32_t) v106;
          uint32_t v109 = v107 + v108;
          int32_t v110 = (int32_t) v109;
          uint32_t v111 = (uint32_t) v110;
          uint32_t v112 = (uint32_t) v9;
          uint32_t v113 = v111 / v112;
          int32_t v114 = (int32_t) v113;
          uint32_t v115 = (uint32_t) v110;
          uint32_t v116 = (uint32_t) v9;
          uint32_t v117 = v115 % v116;
          int32_t v118 = (int32_t) v117;
          uint32_t v119 = (uint32_t) v114;
          uint32_t v120 = (uint32_t) v10;
          uint32_t v121 = v119 / v120;
          int32_t v122 = (int32_t) v121;
          uint32_t v123 = (uint32_t) v122;
          uint32_t v124 = (uint32_t) v7;
          uint32_t v125 = v123 * v124;
          int32_t v126 = (int32_t) v125;
          uint32_t v127 = (uint32_t) v118;
          uint32_t v128 = (uint32_t) v10;
          uint32_t v129 = v127 / v128;
          int32_t v130 = (int32_t) v129;
          uint32_t v131 = (uint32_t) v126;
          uint32_t v132 = (uint32_t) v130;
          uint32_t v133 = v131 + v132;
          int32_t v134 = (int32_t) v133;
          noc_async_write_tile(v134, v38, v94);
          uint32_t v135 = (uint32_t) v94;
          uint32_t v136 = (uint32_t) v35;
          uint32_t v137 = v135 + v136;
          int32_t v138 = (int32_t) v137;
          v92 = v138;
        }
        int32_t v139 = v92;
        v89 = v139;
      }
      noc_async_write_barrier();
      cb_pop_front(v33, v87);
    }
  }
  return;
}

// matmul_kernel__d0i0_d1i1__f01__c0mem_c1mem__host
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "firmware_common.h"
#include "dataflow_api.h"
void kernel_main() {
  int32_t v1 = 8;
  int32_t v2 = 7;
  int32_t v3 = 6;
  int32_t v4 = 5;
  int32_t v5 = 4;
  int32_t v6 = 3;
  int32_t v7 = 2;
  int32_t v8 = 1;
  int32_t v9 = 0;
  ::tt::CB v10 = get_arg_val<uint32_t>(v9);
  int32_t v11 = get_arg_val<uint32_t>(v8);
  int32_t v12 = get_tile_size(v10);
  TensorAccessorArgs v13 = TensorAccessorArgs<0, 0>();
  TensorAccessor v14 = TensorAccessor(v13, v11, v12);
  ::tt::CB v15 = get_arg_val<uint32_t>(v7);
  int32_t v16 = get_arg_val<uint32_t>(v6);
  int32_t v17 = get_tile_size(v15);
  TensorAccessorArgs v18 = TensorAccessorArgs<1, 1>();
  TensorAccessor v19 = TensorAccessor(v18, v16, v17);
  ::tt::CB v20 = get_arg_val<uint32_t>(v5);
  int32_t v21 = get_arg_val<uint32_t>(v4);
  int32_t v22 = get_tile_size(v20);
  TensorAccessorArgs v23 = TensorAccessorArgs<2, 2>();
  TensorAccessor v24 = TensorAccessor(v23, v21, v22);
  int32_t v25 = get_arg_val<uint32_t>(v3);
  int32_t v26 = get_arg_val<uint32_t>(v2);
  int32_t v27 = get_arg_val<uint32_t>(v1);
  constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
  distributed::DeviceLocalBufferConfig dram_config{.page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
  distributed::ReplicatedBufferConfig buffer_config_A{.size = single_tile_size * 64 * 64};
  distributed::ReplicatedBufferConfig buffer_config_B{.size = single_tile_size * 64 * 64};
  distributed::ReplicatedBufferConfig buffer_config_C{.size = single_tile_size * 64 * 64};
  auto dram_buffer_0 = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
  auto dram_buffer_1 = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
  auto dram_buffer_2 = distributed::MeshBuffer::create(buffer_config_C, dram_config, mesh_device.get());
  const auto cb_data_format = tt::DataFormat::Float16_b;
  uint32_t cb_buffer_depth = 2;
  MathFidelity math_fidelity = MathFidelity::HiFi4;
  const uint32_t A_tiles_per_block = 64 * 64;
  const uint32_t B_tiles_per_block = 64 * 64;
  const uint32_t C_tiles_per_block = 64 * 64;
  tt_metal::CreateCircularBuffer(program, all_cores, CircularBufferConfig(A_tiles_per_block * cb_buffer_depth * single_tile_size, {{CBIndex::c_0, cb_data_format}}).set_page_size(CBIndex::c_0, single_tile_size));
  tt_metal::CreateCircularBuffer(program, all_cores, CircularBufferConfig(B_tiles_per_block * cb_buffer_depth * single_tile_size, {{CBIndex::c_1, cb_data_format}}).set_page_size(CBIndex::c_1, single_tile_size));
  tt_metal::CreateCircularBuffer(program, all_cores, CircularBufferConfig(C_tiles_per_block * cb_buffer_depth * single_tile_size, {{CBIndex::c_16, cb_data_format}}).set_page_size(CBIndex::c_16, single_tile_size));
  return;
}

