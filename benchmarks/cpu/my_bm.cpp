#include <benchmark/benchmark.h>
#include "util/mesh.hpp"
#include "pentadsolver.hpp"

template <typename Float> static void BM_PentadSolver(benchmark::State &state) {
  Mesh<Float> mesh(state.range(1),
                   std::vector<int>(3, static_cast<int>(state.range(0))));
  std::vector<Float> x(mesh.x());
  for (auto _ : state) {
    pentadsolver_gpsv_batch(mesh.ds().data(),   // ds
                            mesh.dl().data(),   // dl
                            mesh.d().data(),    // d
                            mesh.du().data(),   // du
                            mesh.dw().data(),   // dw
                            mesh.x().data(),    // x
                            mesh.dims().data(), // t_dims
                            mesh.dims().size(), // t_dims
                            mesh.solve_dim(),   // t_solvedim
                            nullptr);           // t_buffer
    state.PauseTiming();
    mesh.x() = x;
    state.ResumeTiming();
  }
}
// Register the function as a benchmark
constexpr int range_max = 2u << 7;
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
