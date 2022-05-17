#ifndef PENTADSOLVER_CXX_UTIL_HPP_INCLUDED
#define PENTADSOLVER_CXX_UTIL_HPP_INCLUDED

// ----------------------------------------------------------------------------
// Standard Libraries and Headers
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// PentadSolver Headers
// ----------------------------------------------------------------------------
#include "pentadsolver.hpp"

// ============================================================================
// Headerfile Functions
// ============================================================================

// ----------------------------------------------------------------------------
// Buffer size calculation
// ----------------------------------------------------------------------------
/**
 * @brief This function computes the amount of extra memory required for
 * pentadsolver_<?>_gpsv_batch() for the solution of a batch pentadiagonal
 * system of size t_dims, along the axis t_solvedim.
 *
 * @param[in] t_dims      The dimensions of the LHS arrays.
 * @param[in] t_ndims     The length of the t_dims array.
 * @param[in] t_solvedim  The dimension along which the systems are formed.
 *
 * @return size of required buffers in bytes.
 */
template <typename Float>
[[nodiscard]] inline size_t
pentadsolver_gpsv_batch_buffer_size_ext(const int *t_dims, int t_ndims,
                                        int t_solvedim);
template <>
[[nodiscard]] inline size_t
pentadsolver_gpsv_batch_buffer_size_ext<double>(const int *t_dims, int t_ndims,
                                                int t_solvedim) {
  return pentadsolver_D_gpsv_batch_buffer_size_ext(t_dims, t_ndims, t_solvedim);
}

// ----------------------------------------------------------------------------
// Solver functions
// ----------------------------------------------------------------------------
/**
 * @brief This function computes the solution of multiple penta-diagonal systems
 * alng a specified axis.
 *
 *
 * @param[in] ds      Array containing the 2nd lower diagonal.
 * @param[in] dl      Array containing the lower diagonal.
 * @param[in] d       Array containing the main diagonal.
 * @param[in] du      Array containing the upper diagonal.
 * @param[in] dw      Array containing the 2nd upper diagonal.
 * @param[in,out] x   Dense array of RHS as input, dense solution array as
 * output
 * @param[in] t_dims      The dimensions of the LHS arrays.
 * @param[in] t_solvedim  The dimension along which the systems are formed.
 * @param[in] t_buffer  Buffer located by the user, with size at least
 * pentadsolver_D_gpsv_batch_buffer_size_ext(), if t_buffer == nullptr, the
 * function will allocate extra memory
 *
 */
template <typename Float>
inline void pentadsolver_gpsv_batch(const Float *ds, const Float *dl,
                                    const Float *d, const Float *du,
                                    const Float *dw, Float *x,
                                    const int *t_dims, int t_ndim,
                                    int t_solvedim, double *t_buffer);

template <>
inline void pentadsolver_gpsv_batch<double>(const double *ds, const double *dl,
                                            const double *d, const double *du,
                                            const double *dw, double *x,
                                            const int *t_dims, int t_ndim,
                                            int t_solvedim, double *t_buffer) {
  pentadsolver_D_gpsv_batch(ds, dl, d, du, dw, x, t_dims, t_ndim, t_solvedim,
                            t_buffer);
}

#endif /* ifndef PENTADSOLVER_CXX_UTIL_HPP_INCLUDED */
