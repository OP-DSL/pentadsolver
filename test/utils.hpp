#ifndef PENTADSOLVER_TESTS_UTILS_HPP_INCLUDED
#define PENTADSOLVER_TESTS_UTILS_HPP_INCLUDED

#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

template <typename Float> class MeshLoader {
  size_t _solve_dim;
  std::vector<int> _dims;
  std::vector<Float> _ds, _dl, _d, _du, _dw, _x, _u;

public:
  explicit MeshLoader(const std::filesystem::path &file_name);

  [[nodiscard]] size_t solve_dim() const { return _solve_dim; }
  [[nodiscard]] const std::vector<int> &dims() const { return _dims; }
  const std::vector<Float> &ds() const { return _ds; }
  const std::vector<Float> &dl() const { return _dl; }
  const std::vector<Float> &d() const { return _d; }
  const std::vector<Float> &du() const { return _du; }
  const std::vector<Float> &dw() const { return _dw; }
  const std::vector<Float> &x() const { return _x; }
  const std::vector<Float> &u() const { return _u; }
};
/**********************************************************************
 *                          Implementations                           *
 **********************************************************************/

template <typename Float>
inline void load_array(std::ifstream &f, size_t num_elements,
                       std::vector<Float> &array) {
  array.resize(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    // Load with the larger precision, then convert to the specified type
    f >> array[i];
  }
}

template <typename Float>
MeshLoader<Float>::MeshLoader(const std::filesystem::path &file_name)
    : _solve_dim{}, _ds{}, _dl{}, _d{}, _du{}, _dw{}, _u{} {
  std::ifstream f(file_name);
  assert(f.good() && "Couldn't open file");
  size_t num_dims = 0;
  f >> num_dims >> _solve_dim;
  // Load sizes along the different dimensions
  size_t num_elements = 1;
  for (size_t i = 0; i < num_dims; ++i) {
    int size = 0;
    f >> size;
    _dims.push_back(size);
    num_elements *= size;
  }
  // Load arrays
  load_array(f, num_elements, _ds);
  load_array(f, num_elements, _dl);
  load_array(f, num_elements, _d);
  load_array(f, num_elements, _du);
  load_array(f, num_elements, _dw);
  if (std::is_same<Float, double>::value) {
    load_array(f, num_elements, _u);
  } else {
    std::string tmp;
    // Skip the line with the double values
    std::getline(f >> std::ws, tmp);
    load_array(f, num_elements, _u);
  }
}

#endif /* ifndef PENTADSOLVER_TESTS_UTILS_HPP_INCLUDED */
