#ifndef PENTADSOLVER_HPP_INCLUDED
#define PENTADSOLVER_HPP_INCLUDED

// ----------------------------------------------------------------------------
// Standard Libraries and Headers
// ----------------------------------------------------------------------------
#include <cstddef>

// ----------------------------------------------------------------------------
// Buffer size calculation
// ----------------------------------------------------------------------------
/**
 * @brief This function computes the amount of extra memory required for
 * pentadsolver_D_gpsv_batch() for the solution of a batch pentadiagonal system
 * of size t_dims, along the axis t_solvedim.
 *
 * @param[in] t_dims      The dimensions of the LHS arrays.
 * @param[in] t_ndims     The length of the t_dims array.
 * @param[in] t_solvedim  The dimension along which the systems are formed.
 *
 * @return size of required buffers in bytes.
 */
[[nodiscard]] size_t
pentadsolver_D_gpsv_batch_buffer_size_ext(const int *t_dims, int t_ndims,
                                          int t_solvedim);

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
 * @param[in] t_ndims     The length of the t_dims array.
 * @param[in] t_solvedim  The dimension along which the systems are formed.
 * @param[in] t_buffer  Buffer located by the user, with size at least
 * pentadsolver_D_gpsv_batch_buffer_size_ext(), if t_buffer == nullptr, the
 * function will allocate extra memory
 *
 */
void pentadsolver_D_gpsv_batch(const double *ds, const double *dl,
                               const double *d, const double *du,
                               const double *dw, double *x, const int *t_dims,
                               int t_ndim, int t_solvedim, double *t_buffer);

#endif /* ifndef PENTADSOLVER_HPP_INCLUDED */
