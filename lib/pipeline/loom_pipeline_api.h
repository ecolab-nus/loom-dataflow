/// C API for the Loom MLIR pipeline.
///
/// Exposes the combined Materialize → Canonicalize → SymbolDCE → BridgeToOSB
/// → LowerAffineWithAttr → One-Shot Bufferization → Canonicalize → CSE
/// pipeline as a single C-linkage function callable from Python via ctypes.

#ifndef LOOM_PIPELINE_API_H
#define LOOM_PIPELINE_API_H

#ifdef __cplusplus
extern "C" {
#endif

/// Run the full Loom pipeline on an MLIR input file and write the result.
///
/// @param input_mlir_path   Null-terminated path to input MLIR file
///                          (stage 05_after_enumerate_broadcast.mlir).
/// @param block_sizes_json  Null-terminated JSON string mapping variant
///                          function names to symbol assignments, e.g.:
///                          {"matmul__d0i0_d1i0__f01__d_d": {"BM":64,"BN":4096,"BK":512}, ...}
///                          Pass NULL or empty string to use the hardcoded
///                          placeholder solver (backward compat).
/// @param output_mlir_path  Null-terminated path for the output MLIR file.
/// @param error_msg         On failure, set to a malloc'd error string.
///                          Caller must free with loom_free_string().
///                          On success, set to NULL.
/// @return 0 on success, non-zero on error.
int loom_run_full_pipeline(const char *input_mlir_path,
                           const char *block_sizes_json,
                           const char *output_mlir_path,
                           char **error_msg);

/// Free a string returned by loom_run_full_pipeline via error_msg.
void loom_free_string(char *str);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LOOM_PIPELINE_API_H
