#include <cstddef>          // for size_t
#include <functional>       // for multiplies
#include <numeric>          // for accumulate
#include "pentadsolver.hpp" // for pentadsolver_D_gpsv_batch, pentadsolver_...

size_t pentadsolver_D_gpsv_batch_buffer_size_ext(const int *t_dims, int t_ndims,
                                                 int t_solvedim) {
  (void)t_solvedim;
  constexpr size_t number_of_temp_arrays = 4;
  size_t size_of_array =
      std::accumulate(t_dims, t_dims + t_ndims, 1U, std::multiplies<>());
  return number_of_temp_arrays * size_of_array * sizeof(double);
}

void pentadsolver_D_gpsv_batch(const double *ds, const double *dl,
                               const double *d, const double *du,
                               const double *dw, double *x, const int *t_dims,
                               int t_ndim, int t_solvedim, double *t_buffer) {}
