// The GPU dedup tests of node_dedup_gpu_test.cc, run against the simulated
// driver in test/client/sim_cuda instead of a real one: same test bodies, no
// GPU needed. The other binary runs them on real hardware and skips without
// it; this one always runs.
#ifndef DFKV_SIM_CUDA_DIR
#error "DFKV_SIM_CUDA_DIR must name the directory of the simulated libcuda.so.1"
#endif
#define DFKV_GPU_TEST_SIMULATED 1
#include "node_dedup_gpu_test.cc"  // NOLINT(build/include)
