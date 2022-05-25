#ifndef PENTADSOLVER_TESTS_CUDA_UTILS_HPP_INCLUDED
#define PENTADSOLVER_TESTS_CUDA_UTILS_HPP_INCLUDED

#include "utils.hpp"

template <typename Float> class DeviceMeshLoader : public MeshLoader<Float> {
  Float *_ds_d = nullptr;
  Float *_dl_d = nullptr;
  Float *_d_d  = nullptr;
  Float *_du_d = nullptr;
  Float *_dw_d = nullptr;
  Float *_x_d  = nullptr;

public:
  explicit DeviceMeshLoader(const std::filesystem::path &file_name);

  const Float *ds_d() const { return _ds_d; }
  const Float *dl_d() const { return _dl_d; }
  const Float *d_d() const { return _d_d; }
  const Float *du_d() const { return _du_d; }
  const Float *dw_d() const { return _dw_d; }
  const Float *x_d() const { return _x_d; }
  Float *x_d() { return _x_d; }
};

template <typename Float>
DeviceMeshLoader<Float>::DeviceMeshLoader(
    const std::filesystem::path &file_name)
    : MeshLoader<Float>::MeshLoader(file_name) {
  auto init_arr = [](Float **arr_d, const std::vector<Float> &arr_h) {
    cudaMalloc((void **)arr_d, sizeof(Float) * arr_h.size());
    cudaMemcpy(*arr_d, arr_h.data(), sizeof(Float) * arr_h.size(),
               cudaMemcpyHostToDevice);
  };
  init_arr(&_ds_d, this->_ds);
  init_arr(&_dl_d, this->_dl);
  init_arr(&_d_d, this->_d);
  init_arr(&_du_d, this->_du);
  init_arr(&_dw_d, this->_dw);
  init_arr(&_x_d, this->_x);
}

#endif /* ifndef PENTADSOLVER_TESTS_CUDA_UTILS_HPP_INCLUDED */
