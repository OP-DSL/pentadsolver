#include <benchmark/benchmark.h>
#include "util/device_mesh.hpp"
#include "pentadsolver.hpp"

template <typename Float> static void BM_PentadSolver(benchmark::State &state) {
  DeviceMesh<Float> mesh(state.range(1),
                         std::vector<int>(3, static_cast<int>(state.range(0))));
  Flaog *x_d = nullptr;
  cudaMalloc((void **)&x_d, sizeof(Float) * mesh.x().size());
  cudaMemcpy(x_d, mesh.x_d(), sizeof(Float) * mesh.x().size(),
             cudaMemcpyDeviceToDevice);
  for (auto _ : state) {
    pentadsolver_gpsv_batch(mesh.ds_d(),        // ds
                            mesh.dl_d(),        // dl
                            mesh.d_d(),         // d
                            mesh.du_d(),        // du
                            mesh.dw_d(),        // dw
                            mesh.x_d(),         // x
                            mesh.dims().data(), // t_dims
                            mesh.dims().size(), // t_dims
                            mesh.solve_dim(),   // t_solvedim
                            nullptr);           // t_buffer
    state.PauseTiming();
    cudaMemcpy(mesh.x_d(), x_d, sizeof(Float) * mesh.x().size(),
               cudaMemcpyDeviceToDevice);
    state.ResumeTiming();
  }
}
// Register the function as a benchmark
constexpr int range_max = 2 << 7;
BENCHMARK_TEMPLATE(BM_PentadSolver, float)
    ->RangeMultiplier(2)
    ->Ranges({{32, range_max}, {1, 1}, {0, 0}});
BENCHMARK_TEMPLATE(BM_PentadSolver, double)
    ->RangeMultiplier(2)
    ->Ranges({{32, range_max}, {1, 1}, {0, 0}});
BENCHMARK_TEMPLATE(BM_PentadSolver, float)
    ->RangeMultiplier(2)
    ->Ranges({{32, range_max}, {2, 2}, {0, 1}});
BENCHMARK_TEMPLATE(BM_PentadSolver, double)
    ->RangeMultiplier(2)
    ->Ranges({{32, range_max}, {2, 2}, {0, 1}});
BENCHMARK_TEMPLATE(BM_PentadSolver, float)
    ->RangeMultiplier(2)
    ->Ranges({{32, range_max}, {3, 3}, {0, 2}});
BENCHMARK_TEMPLATE(BM_PentadSolver, double)
    ->RangeMultiplier(2)
    ->Ranges({{32, range_max}, {3, 3}, {0, 2}});

BENCHMARK_MAIN();
