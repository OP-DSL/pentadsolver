#include <mpi.h>                   // for MPI_Request, MPI_REQUEST_NULL
#include <cstddef>                 // for size_t
#include <array>                   // for array
#include <cassert>                 // for assert
#include <cmath>                   // for ceil, log2
#include <functional>              // for multiplies
#include <numeric>                 // for accumulate
#include <type_traits>             // for is_same
#include <vector>                  // for vector
#include "pentadsolver.hpp"        // for pentadsolver_gpsv_batch, pentadso...
#include "pentadsolver_handle.hpp" // for pentadsolver_handle_implementation

namespace {
template <typename REAL>
const MPI_Datatype mpi_datatype =
    std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;
}

template <typename Float>
void gpsv_forward_x(const Float *ds, const Float *dl, const Float *d,
                    const Float *du, const Float *dw, Float *x, Float *dss,
                    Float *dll, Float *duu, Float *dww, Float *comm_buf,
                    size_t t_sys_size) {
  assert(t_sys_size > 2);
  // row 0:
  Float s0 = ds[0]; // FIXME might segfault
  Float l0 = dl[0]; // FIXME might segfault
  Float d0 = d[0];
  Float u0 = du[0];
  Float w0 = dw[0];
  Float x0 = x[0];
  // row 1: norm, shift u0, w0
  x[1]   = x[1] / d[1];
  dss[1] = ds[1] / d[1]; // FIXME might segfault
  dll[1] = dl[1] / d[1];
  duu[1] = du[1] / d[1];
  dww[1] = dw[1] / d[1]; // FIXME might segfault
  x0     = x0 - u0 * x[1];
  l0     = l0 - u0 * dss[1];
  d0     = d0 - u0 * dll[1];
  w0     = 00 - u0 * dww[1];
  u0     = w0 - u0 * duu[1];
  // row 2: shift {s, l} with row 2, norm, shift u0, w0
  Float li  = dl[2];
  Float ddi = d[2] - li * duu[1];
  x[2]      = (x[2] - li * x[1]) / ddi;
  dss[2]    = (-li * dss[1]) / ddi;
  dll[2]    = (ds[2] - li * dll[1]) / ddi;
  duu[2]    = (du[2] - li * dww[1]) / ddi; // FIXME might segfault
  dww[2]    = (dw[2]) / ddi;               // FIXME might segfault
  if (3 < t_sys_size) {                    // check if row 2 is the last
    x0 = x0 - u0 * x[2];
    l0 = l0 - u0 * dss[2];
    d0 = d0 - u0 * dll[2];
    w0 = u0 * dww[2];
    u0 = w0 - u0 * duu[2];
    // from row 3 till t_sys_size -1:
    // shift {s, l} with row i-2 and i-1, norm, shift u0, w0
    for (int i = 3; i < t_sys_size - 1; ++i) {
      Float si = ds[i];
      li       = dl[i] - si * duu[i - 2];
      ddi      = d[i] - si * dww[i - 2] - li * duu[i - 1];
      x[i]     = (-si * x[i - 2] - li * x[i - 1]) / ddi;
      dss[i]   = (-si * dss[i - 2] - li * dss[i - 1]) / ddi;
      dll[i]   = (-si * dll[i - 2] - li * dll[i - 1]) / ddi;
      duu[i]   = (du[i] - li * dww[i - 1]) / ddi;
      dww[i]   = (dw[i]) / ddi;
      // shift u0, w0
      x0 = x0 - u0 * x[i];
      l0 = l0 - u0 * dss[i];
      d0 = d0 - u0 * dll[i];
      w0 = u0 * dww[i]; // FIXME might segfault
      u0 = w0 - u0 * duu[i];
    }
    // last row shift {s, l} with row i-2 and i-1, norm
    size_t i = t_sys_size - 1;
    Float si = ds[i];
    li       = dl[i] - si * duu[i - 2];
    ddi      = d[i] - si * dww[i - 2] - li * duu[i - 1];
    x[i]     = (-si * x[i - 2] - li * x[i - 1]) / ddi;
    dss[i]   = (-si * dss[i - 2] - li * dss[i - 1]) / ddi;
    dll[i]   = (-si * dll[i - 2] - li * dll[i - 1]) / ddi;
    duu[i]   = (du[i] - li * dww[i - 1]) / ddi; // FIXME might segfault
    dww[i]   = (dw[i]) / ddi;                   // FIXME might segfault
  }
  // norm first row
  s0 = s0 / d0;
  l0 = l0 / d0;
  u0 = u0 / d0;
  w0 = w0 / d0;
  x0 = x0 / d0;
  // to combuf
  comm_buf[0] = s0;
  comm_buf[1] = dss[t_sys_size - 1];
  comm_buf[2] = l0;
  comm_buf[3] = dll[t_sys_size - 1];
  comm_buf[4] = u0;
  comm_buf[5] = duu[t_sys_size - 1]; // NOLINT
  comm_buf[6] = w0;                  // NOLINT
  comm_buf[7] = dww[t_sys_size - 1]; // NOLINT
  comm_buf[8] = x0;                  // NOLINT
  comm_buf[9] = x[t_sys_size - 1];   // NOLINT
}

template <typename Float>
void gpsv_forward_batched_x(const Float *ds, const Float *dl, const Float *d,
                            const Float *du, const Float *dw, Float *x,
                            Float *dss, Float *dll, Float *duu, Float *dww,
                            Float *comm_buf, size_t t_n_sys, int t_sys_size) {
  // #pragma omp parallel for
  for (int sys_id = 0; sys_id < t_n_sys; ++sys_id) {
    int sys_start              = sys_id * t_sys_size;
    constexpr int nvar_per_sys = 10;
    int sys_comm_start         = sys_id * nvar_per_sys;
    gpsv_forward_x(ds + sys_start, dl + sys_start, d + sys_start,
                   du + sys_start, dw + sys_start, x + sys_start,
                   dss + sys_start, dll + sys_start, duu + sys_start,
                   dww + sys_start, comm_buf + sys_comm_start, t_sys_size);
  }
}

template <typename Float>
void gpsv_backward_x(const Float *dss, const Float *dll, const Float *duu,
                     const Float *dww, Float xm1, Float x0, Float x_last,
                     Float xp1, Float *x, int t_sys_size) {
  x[0]              = x0;
  x[t_sys_size - 1] = x_last;
  Float x_i_p2      = xp1;
  for (int i = t_sys_size - 2; i > 0; --i) {
    x[i] =
        x[i] - dww[i] * x_i_p2 - duu[i] - x[i + 1] - dll[i] * x0 - dss[i] * xm1;
    x_i_p2 = x[i + 1];
  }
}

template <typename Float>
void gpsv_backward_batched_x(const Float *dss, const Float *dll,
                             const Float *duu, const Float *dww, Float *x,
                             Float *comm_buf, Float *rcvbufL, Float *rcvbufR,
                             size_t t_n_sys, int t_sys_size) {
#pragma omp parallel for
  for (int sys_id = 0; sys_id < t_n_sys; ++sys_id) {
    int sys_start              = sys_id * t_sys_size;
    constexpr int nvar_per_sys = 10;
    int sys_comm_x_start       = (sys_id + 1) * nvar_per_sys - 2;
    gpsv_backward_x(dss + sys_start, dll + sys_start, duu + sys_start,
                    dww + sys_start, rcvbufL[sys_id],
                    comm_buf[sys_comm_x_start], comm_buf[sys_comm_x_start + 1],
                    rcvbufR[sys_id], x + sys_start, t_sys_size);
  }
}

// Solve reduced system with PCR algorithm.
// sndbuf holds the last and first row of each system after the forward pass as
// an input and the solutions at the x positions as an output.
// sndbuf stores each system packed as: s0, s1, l0, l1, u0, u1, w0, w1, x0, x1
// For each node for each system the first and last row of
// create the reduced system, d_i = 1 everywhere.
// rcvbuf must be an array with size at least 2 * 10 * n_sys.
// rcvbuf[0:nsys], rcvbuf[nsys+1:2*nsys] will hold the solution xm1, xp1
// on exit, respectively
template <typename Float>
inline void solve_reduced_pcr(pentadsolver_handle_t params, Float *rcvbuf,
                              Float *sndbuf, int t_solvedim, int t_n_sys) {
  int rank                   = params->mpi_coords[t_solvedim];
  int nproc                  = params->num_mpi_procs[t_solvedim];
  constexpr int tag          = 1242;
  constexpr int nvar_per_sys = 10;
  Float *rcvbufL             = rcvbuf;
  Float *rcvbufR             = rcvbuf + t_n_sys * nvar_per_sys;

  int P = std::ceil(std::log2((double)params->num_mpi_procs[t_solvedim]));
  int s = 1;
  // perform pcr
  // loop for for comm
  for (int p = 0; p < P; ++p) {
    // rank diff to communicate with
    int leftrank                    = rank - s;
    int rightrank                   = rank + s;
    std::array<MPI_Request, 4> reqs = {
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
    };
    // Get the minus elements
    if (leftrank >= 0) {
      // send recv
      MPI_Isend(sndbuf, t_n_sys * nvar_per_sys, mpi_datatype<Float>, leftrank,
                tag, params->communicators[t_solvedim], &reqs[2]);
      MPI_Irecv(rcvbufL, t_n_sys * nvar_per_sys, mpi_datatype<Float>, leftrank,
                tag, params->communicators[t_solvedim], &reqs[0]);
    }

    // Get the plus elements
    if (rightrank < nproc) {
      // send recv
      MPI_Isend(sndbuf, t_n_sys * nvar_per_sys, mpi_datatype<Float>, rightrank,
                tag, params->communicators[t_solvedim], &reqs[3]);
      MPI_Irecv(rcvbufR, t_n_sys * nvar_per_sys, mpi_datatype<Float>, rightrank,
                tag, params->communicators[t_solvedim], &reqs[1]);
    }

    // Wait for receives to finish
    MPI_Waitall(4, reqs.data(), MPI_STATUS_IGNORE);

    // PCR algorithm
#pragma omp parallel for
    for (int id = 0; id < t_n_sys; id++) {
      // s l 1 0 u w         -> s l 1 0 u w
      // s l 0 1 u w         -> s l 0 1 u w
      // ------------------- -> -------------------
      //     s l 1 0 u w     -> s l     1 0     u w
      //     s l 0 1 u w     -> s l     0 1     u w
      // ------------------- -> -------------------
      //         s l 1 0 u w ->         s l 1 0 u w
      //         s l 0 1 u w ->         s l 0 1 u w
      int sidx             = id * nvar_per_sys;
      constexpr int s0_idx = 0;
      constexpr int s1_idx = 1;
      constexpr int l0_idx = 2;
      constexpr int l1_idx = 3;
      constexpr int u0_idx = 4;
      constexpr int u1_idx = 5;
      constexpr int w0_idx = 6;
      constexpr int w1_idx = 7;
      constexpr int x0_idx = 8;
      constexpr int x1_idx = 9;
      Float sm2            = 0.0;
      Float sm1            = 0.0;
      Float lm2            = 0.0;
      Float lm1            = 0.0;
      Float um2            = 0.0;
      Float um1            = 0.0;
      Float wm2            = 0.0;
      Float wm1            = 0.0;
      Float xm2            = 0.0;
      Float xm1            = 0.0;
      Float sp1            = 0.0;
      Float sp2            = 0.0;
      Float lp1            = 0.0;
      Float lp2            = 0.0;
      Float up1            = 0.0;
      Float up2            = 0.0;
      Float wp1            = 0.0;
      Float wp2            = 0.0;
      Float xp1            = 0.0;
      Float xp2            = 0.0;
      if (leftrank >= 0) {
        sm2 = rcvbufL[sidx + s0_idx];
        sm1 = rcvbufL[sidx + s1_idx];
        lm2 = rcvbufL[sidx + l0_idx];
        lm1 = rcvbufL[sidx + l1_idx];
        um2 = rcvbufL[sidx + u0_idx];
        um1 = rcvbufL[sidx + u1_idx];
        wm2 = rcvbufL[sidx + w0_idx];
        wm1 = rcvbufL[sidx + w1_idx];
        xm2 = rcvbufL[sidx + x0_idx];
        xm1 = rcvbufL[sidx + x1_idx];
      }
      if (rightrank < nproc) {
        sp1 = rcvbufR[sidx + s0_idx];
        sp2 = rcvbufR[sidx + s1_idx];
        lp1 = rcvbufR[sidx + l0_idx];
        lp2 = rcvbufR[sidx + l1_idx];
        up1 = rcvbufR[sidx + u0_idx];
        up2 = rcvbufR[sidx + u1_idx];
        wp1 = rcvbufR[sidx + w0_idx];
        wp2 = rcvbufR[sidx + w1_idx];
        xp1 = rcvbufR[sidx + x0_idx];
        xp2 = rcvbufR[sidx + x1_idx];
      }
      Float s0 = sndbuf[sidx + s0_idx];
      Float s1 = sndbuf[sidx + s1_idx];
      Float l0 = sndbuf[sidx + l0_idx];
      Float l1 = sndbuf[sidx + l1_idx];
      Float u0 = sndbuf[sidx + u0_idx];
      Float u1 = sndbuf[sidx + u1_idx];
      Float w0 = sndbuf[sidx + w0_idx];
      Float w1 = sndbuf[sidx + w1_idx];
      Float x0 = sndbuf[sidx + x0_idx];
      Float x1 = sndbuf[sidx + x1_idx];
      // shift by one:
      Float d0              = 1 - l0 * um1 - s0 * um2 - u0 * sp1 - w0 * sp2;
      Float tmp0            = -l0 * wm1 - s0 * wm2 - u0 * lp1 - w0 * lp2;
      Float d1              = 1 - l1 * wm1 - s1 * wm2 - u1 * lp1 - w1 * lp2;
      Float tmp1            = -l1 * um1 - s1 * um2 - u1 * sp1 - w1 * sp2;
      sndbuf[sidx + s0_idx] = -l0 * sm1 - s0 * sm2;                        // s0
      sndbuf[sidx + l0_idx] = -l0 * lm1 - s0 * lm2;                        // l0
      sndbuf[sidx + u0_idx] = -u0 * up1 - w0 * up2;                        // u0
      sndbuf[sidx + w0_idx] = -u0 * wp1 - w0 * wp2;                        // w0
      sndbuf[sidx + x0_idx] -= -u0 * xp1 - w0 * xp2 - l0 * xm1 - s0 * xm2; // x0
      sndbuf[sidx + s1_idx] = -l1 * sm1 - s1 * sm2;                        // s1
      sndbuf[sidx + l1_idx] = -l1 * lm1 - s1 * lm2;                        // l1
      sndbuf[sidx + u1_idx] = -u1 * up1 - w1 * up2;                        // u1
      sndbuf[sidx + w1_idx] = -u1 * wp1 - w1 * wp2;                        // w1
      sndbuf[sidx + x1_idx] -= -u1 * xp1 - w1 * xp2 - l1 * xm1 - s1 * xm2; // x1
      // zero out tmp values
      //     s l d T u w     -> s l 1 0 u w
      //     s l T d u w     -> s l T d u w
      Float coeff = tmp0 / d1;
      d0          = d0 - coeff * tmp1;
      sndbuf[sidx + s0_idx] =
          (sndbuf[sidx + s0_idx] - sndbuf[sidx + s1_idx] * coeff) / d0; // s0
      sndbuf[sidx + l0_idx] =
          (sndbuf[sidx + l0_idx] - sndbuf[sidx + l1_idx] * coeff) / d0; // l0
      sndbuf[sidx + u0_idx] =
          (sndbuf[sidx + u0_idx] - sndbuf[sidx + u1_idx] * coeff) / d0; // u0
      sndbuf[sidx + w0_idx] =
          (sndbuf[sidx + w0_idx] - sndbuf[sidx + w1_idx] * coeff) / d0; // w0
      sndbuf[sidx + x0_idx] =
          (sndbuf[sidx + x0_idx] - sndbuf[sidx + x1_idx] * coeff) / d0; // x0
      //     s l 1 0 u w     -> s l 1 0 u w
      //     s l T d u w     -> s l 0 1 u w
      sndbuf[sidx + s1_idx] =
          (sndbuf[sidx + s1_idx] - sndbuf[sidx + s0_idx] * tmp1) / d1; // s1
      sndbuf[sidx + l1_idx] =
          (sndbuf[sidx + l1_idx] - sndbuf[sidx + l0_idx] * tmp1) / d1; // l1
      sndbuf[sidx + u1_idx] =
          (sndbuf[sidx + u1_idx] - sndbuf[sidx + u0_idx] * tmp1) / d1; // u1
      sndbuf[sidx + w1_idx] =
          (sndbuf[sidx + w1_idx] - sndbuf[sidx + w0_idx] * tmp1) / d1; // w1
      sndbuf[sidx + x1_idx] =
          (sndbuf[sidx + x1_idx] - sndbuf[sidx + x0_idx] * tmp1) / d1; // x1
    }

    // done
    s = s << 1; // NOLINT
  }
  // pack new sndbuf:
#pragma omp parallel for
  for (int i = 0; i < t_n_sys; ++i) {
    constexpr int x0_idx = 8;
    constexpr int x1_idx = 9;
    rcvbufR[i]           = sndbuf[i * nvar_per_sys + x0_idx];
    rcvbufR[t_n_sys + i] = sndbuf[i * nvar_per_sys + x1_idx];
  }
  // send 1-1 row left and right
  {
    int leftrank                    = rank - 1;
    int rightrank                   = rank + 1;
    std::array<MPI_Request, 4> reqs = {
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
    };
    // Get the minus elements
    if (leftrank >= 0) {
      // send recv
      MPI_Isend(rcvbufR, t_n_sys, mpi_datatype<Float>, leftrank, tag,
                params->communicators[t_solvedim], &reqs[2]);
      MPI_Irecv(rcvbufL, t_n_sys, mpi_datatype<Float>, leftrank, tag,
                params->communicators[t_solvedim], &reqs[0]);
    }

    // Get the plus elements
    if (rightrank < nproc) {
      // send recv
      MPI_Isend(rcvbufR + t_n_sys, t_n_sys, mpi_datatype<Float>, rightrank, tag,
                params->communicators[t_solvedim], &reqs[3]);
      MPI_Irecv(rcvbufL + t_n_sys, t_n_sys, mpi_datatype<Float>, rightrank, tag,
                params->communicators[t_solvedim], &reqs[1]);
    }

    // Wait for receives to finish
    MPI_Waitall(4, reqs.data(), MPI_STATUS_IGNORE);
  }
}

template <typename Float>
void gpsv_batched_forward(const Float *ds, const Float *dl, const Float *d,
                          const Float *du, const Float *dw, Float *x,
                          Float *dss, Float *dll, Float *duu, Float *dww,
                          Float *comm_buf, const int *t_dims, size_t t_n_sys,
                          int t_solvedim) {
  if (t_solvedim == 0) {
    gpsv_forward_batched_x(ds, dl, d, du, dw, x, dss, dll, duu, dww, comm_buf,
                           t_n_sys, t_dims[t_solvedim]);
    // } else if (t_solvedim == t_ndims - 1) {
    //   pentadsolver_gpsv_batch_outermost(ds, dl, d, du, dw, x, t_dims,
    //   t_ndims,
    //                                     t_buffer)
    // } else {
    //   pentadsolver_gpsv_batch_middle(ds, dl, d, du, dw, x, t_dims, t_ndims,
    //                                  t_solvedim, t_buffer);
  }
}

template <typename Float>
void gpsv_backward_batched(const Float *dss, const Float *dll, const Float *duu,
                           const Float *dww, Float *x, Float *comm_buf,
                           Float *rcvbufL, Float *rcvbufR, const int *t_dims,
                           size_t t_n_sys, int t_solvedim) {
  if (t_solvedim == 0) {
    gpsv_backward_batched_x(dss, dll, duu, dww, x, comm_buf, rcvbufL, rcvbufR,
                            t_n_sys, t_dims[t_solvedim]);
    // } else if (t_solvedim == t_ndims - 1) {
    //   pentadsolver_gpsv_batch_outermost(ds, dl, d, du, dw, x, t_dims,
    //   t_ndims,
    //                                     t_buffer)
    // } else {
    //   pentadsolver_gpsv_batch_middle(ds, dl, d, du, dw, x, t_dims, t_ndims,
    //                                  t_solvedim, t_buffer);
  }
}

template <typename Float>
void pentadsolver_gpsv_batch(pentadsolver_handle_t handle, const Float *ds,
                             const Float *dl, const Float *d, const Float *du,
                             const Float *dw, Float *x, const int *t_dims,
                             size_t t_ndims, int t_solvedim, void *t_buffer) {
  size_t buff_size =
      std::accumulate(t_dims, t_dims + t_ndims, 1, std::multiplies<>());
  size_t n_sys = buff_size / t_dims[t_solvedim];
  Float *dss   = reinterpret_cast<Float *>(t_buffer) + buff_size * 0;
  Float *dll   = reinterpret_cast<Float *>(t_buffer) + buff_size * 1;
  Float *duu   = reinterpret_cast<Float *>(t_buffer) + buff_size * 2;
  Float *dww   = reinterpret_cast<Float *>(t_buffer) + buff_size * 3;
  constexpr int reduced_size_elem = 10; // 2 row per node 5 value per row
  Float *sndbuf                   = dww + buff_size;
  Float *rcvbuf                   = sndbuf + n_sys * reduced_size_elem;
  gpsv_batched_forward(ds, dl, d, du, dw, x, dss, dll, duu, dww, sndbuf, t_dims,
                       n_sys, t_solvedim);
  solve_reduced_pcr(handle, rcvbuf, sndbuf, t_solvedim, n_sys);
  gpsv_backward_batched(dss, dll, duu, dww, x, sndbuf, rcvbuf, rcvbuf + n_sys,
                        t_dims, n_sys, t_solvedim);
}

// ----------------------------------------------------------------------------
// Pentadsolver context functions
// ----------------------------------------------------------------------------

void pentadsolver_create(pentadsolver_handle_t *handle, void *communicator,
                         int ndims, const int *num_procs) {

  // NOLINTNEXTLINE
  *handle = new pentadsolver_handle_implementation(
      *reinterpret_cast<MPI_Comm *>(communicator), ndims, num_procs);
}
void pentadsolver_destroy(pentadsolver_handle_t *handle) {
  // NOLINTNEXTLINE
  delete *handle;
}

// ----------------------------------------------------------------------------
// Buffer size calculation
// ----------------------------------------------------------------------------

template <typename Float>
[[nodiscard]] size_t pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t /*handle*/, const Float * /*ds*/,
    const Float * /*dl*/, const Float * /*d*/, const Float * /*du*/,
    const Float * /*dw*/, const Float * /*x*/, const int *t_dims,
    size_t t_ndims, int t_solvedim) {
  // Communication would need 3 buffers: 1 send 2 receive
  // Each buffer holds 2 rows for each system with d == 1 -> 5 elem per row
  // size = n_sys * 2 * 5 * 3
  // Intermediate results need 4 buffers: ds, dl, du, dw
  // each with the same size as the originals
  size_t buff_size =
      std::accumulate(t_dims, t_dims + t_ndims, 1, std::multiplies<>());
  size_t n_sys                       = buff_size / t_dims[t_solvedim];
  constexpr int num_comm_bufs        = 3;
  constexpr int nvar_per_sys         = 10;
  constexpr int num_intermediate_buf = 4;

  return (n_sys * nvar_per_sys * num_comm_bufs +
          buff_size * num_intermediate_buf) *
         sizeof(Float);
}

// ----------------------------------------------------------------------------
// Adapter function implementations
// ----------------------------------------------------------------------------

size_t pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const double *ds, const double *dl,
    const double *d, const double *du, const double *dw, const double *x,
    const int *t_dims, int t_ndim, int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(
      handle, ds, dl, d, du, dw, x, t_dims, static_cast<size_t>(t_ndim),
      t_solvedim);
}

size_t pentadsolver_D_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const double *ds, const double *dl,
    const double *d, const double *du, const double *dw, double *x,
    const int *t_dims, int t_ndim, int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(handle, ds, dl, d, du, dw, x,
                                                 t_dims, t_ndim, t_solvedim);
}

size_t pentadsolver_gpsv_batch_buffer_size_ext(pentadsolver_handle_t handle,
                                               const float *ds, const float *dl,
                                               const float *d, const float *du,
                                               const float *dw, const float *x,
                                               const int *t_dims, int t_ndim,
                                               int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(
      handle, ds, dl, d, du, dw, x, t_dims, static_cast<size_t>(t_ndim),
      t_solvedim);
}

size_t pentadsolver_S_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const float *ds, const float *dl,
    const float *d, const float *du, const float *dw, float *x,
    const int *t_dims, int t_ndim, int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(handle, ds, dl, d, du, dw, x,
                                                 t_dims, t_ndim, t_solvedim);
}

void pentadsolver_gpsv_batch(pentadsolver_handle_t handle, const double *ds,
                             const double *dl, const double *d,
                             const double *du, const double *dw, double *x,
                             const int *t_dims, int t_ndim, int t_solvedim,
                             void *t_buffer) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims,
                          static_cast<size_t>(t_ndim), t_solvedim, t_buffer);
}

void pentadsolver_D_gpsv_batch(pentadsolver_handle_t handle, const double *ds,
                               const double *dl, const double *d,
                               const double *du, const double *dw, double *x,
                               const int *t_dims, int t_ndim, int t_solvedim,
                               void *t_buffer) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims, t_ndim,
                          t_solvedim, t_buffer);
}

void pentadsolver_gpsv_batch(pentadsolver_handle_t handle, const float *ds,
                             const float *dl, const float *d, const float *du,
                             const float *dw, float *x, const int *t_dims,
                             int t_ndim, int t_solvedim, void *t_buffer) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims,
                          static_cast<size_t>(t_ndim), t_solvedim, t_buffer);
}

void pentadsolver_S_gpsv_batch(pentadsolver_handle_t handle, const float *ds,
                               const float *dl, const float *d, const float *du,
                               const float *dw, float *x, const int *t_dims,
                               int t_ndim, int t_solvedim, void *t_buffer) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims, t_ndim,
                          t_solvedim, t_buffer);
}
