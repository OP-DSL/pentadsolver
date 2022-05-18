#include <catch2/catch.hpp> // for Section, INTERNAL_CATCH_NOINTERNAL_...
#include <filesystem>       // for path
#include <vector>           // for allocator, vector
#include "catch_utils.hpp"  // for require_allclose
#include "pentadsolver.hpp" // for pentadsolver_gpsv_batch
#include "utils.hpp"        // for MeshLoader

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

TEMPLATE_TEST_CASE("x_solve: batch small", "[small]", double) { // NOLINT
  SECTION("ndims: 1") {
    test_from_file<TestType>("files/one_dim_small_solve0");
  }
  SECTION("ndims: 2") {
    test_from_file<TestType>("files/two_dim_small_solve0");
  }
}

TEMPLATE_TEST_CASE("y_solve: batch small", "[small]", double) { // NOLINT
  SECTION("ndims: 2") {
    test_from_file<TestType>("files/two_dim_small_solve1");
  }
}

TEMPLATE_TEST_CASE("x_solve: batch large", "[large]", double) { // NOLINT
  SECTION("ndims: 1") {
    test_from_file<TestType>("files/one_dim_large_solve0");
  }
  SECTION("ndims: 2") {
    test_from_file<TestType>("files/two_dim_large_solve0");
  }
  SECTION("ndims: 3") {
    test_from_file<TestType>("files/three_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE("y_solve: batch large", "[large]", double) { // NOLINT
  SECTION("ndims: 2") {
    test_from_file<TestType>("files/two_dim_large_solve1");
  }
  SECTION("ndims: 3") {
    test_from_file<TestType>("files/three_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE("z_solve: batch large", "[large]", double) { // NOLINT
  SECTION("ndims: 3") {
    test_from_file<TestType>("files/three_dim_large_solve2");
  }
}

