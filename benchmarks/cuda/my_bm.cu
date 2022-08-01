#include <benchmark/benchmark.h>
#include "util/device_mesh.hpp"
#include "pentadsolver.hpp"

template <typename Float> static void BM_PentadSolver(benchmark::State &state) {
  DeviceMesh<Float> mesh(
      state.range(2),
      std::vector<int>(state.range(1), static_cast<int>(state.range(0))));
  Float *x_d = nullptr;
  cudaMalloc((void **)&x_d, sizeof(Float) * mesh.x().size());
  cudaMemcpy(x_d, mesh.x_d(), sizeof(Float) * mesh.x().size(),
             cudaMemcpyDeviceToDevice);
  pentadsolver_handle_t handle{};
  pentadsolver_create(&handle, nullptr, 0, nullptr);
  for (auto _ : state) {
    pentadsolver_gpsv_batch(handle,             // context
                            mesh.ds().data(),   // ds
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
constexpr int range_min = 32;
constexpr int range_max = 2U << 7U;
// NOLINTNEXTLINE(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-owning-memory)
BENCHMARK_TEMPLATE(BM_PentadSolver, float)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {1, 1}, {0, 0}});
// NOLINTNEXTLINE(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-owning-memory)
BENCHMARK_TEMPLATE(BM_PentadSolver, double)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {1, 1}, {0, 0}});
// NOLINTNEXTLINE(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-owning-memory)
BENCHMARK_TEMPLATE(BM_PentadSolver, float)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {2, 2}, {0, 1}});
// NOLINTNEXTLINE(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-owning-memory)
BENCHMARK_TEMPLATE(BM_PentadSolver, double)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {2, 2}, {0, 1}});
// NOLINTNEXTLINE(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-owning-memory)
BENCHMARK_TEMPLATE(BM_PentadSolver, float)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {3, 3}, {0, 2}});
// NOLINTNEXTLINE(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-owning-memory)
BENCHMARK_TEMPLATE(BM_PentadSolver, double)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {3, 3}, {0, 2}});

BENCHMARK_MAIN();
