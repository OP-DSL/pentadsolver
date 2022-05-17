#include <catch2/catch.hpp>       // for Section, INTERNAL_CATCH_NOINTERNAL_...
#include <filesystem>             // for path
#include <vector>                 // for allocator, vector
#include "catch_utils.hpp"        // for require_allclose
#include "pentadsolver_util.hpp"  // for pentadsolver_gpsv_batch
#include "utils.hpp"              // for MeshLoader

template <typename Float>
void test_from_file(const std::filesystem::path &file_name) {
  MeshLoader<Float> mesh(file_name);
  std::vector<Float> x(mesh.x());

  pentadsolver_gpsv_batch(mesh.ds().data(),   // ds
                          mesh.dl().data(),   // dl
                          mesh.d().data(),    // d
                          mesh.du().data(),   // du
                          mesh.dw().data(),   // dw
                          x.data(),           // x
                          mesh.dims().data(), // t_dims
                          mesh.dims().size(), // t_dims
                          mesh.solve_dim(),   // t_solvedim
                          nullptr);           // t_buffer

  require_allclose(mesh.u(), x);
}

TEMPLATE_TEST_CASE("cpu: batch small", "[small]", double) { // NOLINT
  SECTION("ndims: 1") { test_from_file<TestType>("files/one_dim_small_solve0"); }
  SECTION("ndims: 2") {
    SECTION("solvedim: 0") {
      test_from_file<TestType>("files/two_dim_small_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file<TestType>("files/two_dim_small_solve1");
    }
  }
}

TEMPLATE_TEST_CASE("cpu: strided batch large", "[large]", double) { // NOLINT
  SECTION("ndims: 1") { test_from_file<TestType>("files/one_dim_large_solve0"); }
  SECTION("ndims: 2") {
    SECTION("solvedim: 0") {
      test_from_file<TestType>("files/two_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file<TestType>("files/two_dim_large_solve1");
    }
  }
  SECTION("ndims: 3") {
    SECTION("solvedim: 0") {
      test_from_file<TestType>("files/three_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file<TestType>("files/three_dim_large_solve1");
    }
    SECTION("solvedim: 2") {
      test_from_file<TestType>("files/three_dim_large_solve2");
    }
  }
}

