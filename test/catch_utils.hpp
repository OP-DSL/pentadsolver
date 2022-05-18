#ifndef CATCH_UTILS_HPP_INCLUDED
#define CATCH_UTILS_HPP_INCLUDED
#include <catch2/catch.hpp>

#include <vector>

template <typename Float>
void require_allclose(const std::vector<Float> &expected,
                      const std::vector<Float> &actual,
                      std::string_view tag = "") {
  REQUIRE(expected.size() == actual.size());
  CAPTURE(tag);
  Catch::StringMaker<Float>::precision =
      std::numeric_limits<Float>::max_digits10;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double diff = std::abs(static_cast<double>(expected[i]) - actual[i]);
    CAPTURE(i, expected[i], actual[i], expected.size(), diff);
    constexpr double tolerance =
        1e7; // FIXME check error bound, and reference solution
    constexpr double abs_tolerance =
        1e-12; // FIXME check error bound, and reference solution
    REQUIRE_THAT(
        actual[i],
        Catch::Matchers::WithinRel(
            expected[i], std::numeric_limits<Float>::epsilon() * tolerance) ||
            Catch::Matchers::WithinAbs(expected[i], abs_tolerance));
  }
}

#endif /* ifndef CATCH_UTILS_HPP_INCLUDED */
