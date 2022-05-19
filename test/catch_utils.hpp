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

    constexpr Float abs_tolerance =
        std::is_same_v<Float, double> ? 1e-15 : 1e-6;
    constexpr Float rel_tolerance =
        std::is_same_v<Float, double> ? 1e-13 : 1e-5;
    REQUIRE_THAT(actual[i],
                 Catch::Matchers::WithinRel(expected[i], rel_tolerance) ||
                     Catch::Matchers::WithinAbs(expected[i], abs_tolerance));
  }
}

#endif /* ifndef CATCH_UTILS_HPP_INCLUDED */
