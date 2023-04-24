#include <array>
#include <cctype>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <mpi.h>
#include <thread>
#include <unistd.h>

#ifndef PENTAD_PERF_CUDA
#include "pentadsolver.hpp"        // for pentadsolver_destroy, pentadsolve...
#include "pentadsolver_handle.hpp" // for pentadsolver_handle_implementation
#include "util/mesh.hpp"           // for Mesh

template <typename Float> using MyMesh = Mesh<Float>;

template <typename Float>
void run_solver(pentadsolver_handle_t handle, MyMesh<Float> &mesh,
                int num_iters) {
  std::vector<Float> x(mesh.x());

  size_t extent =
      pentadsolver_gpsv_batch_buffer_size_ext(handle,             // context
                                              mesh.ds().data(),   // ds
                                              mesh.dl().data(),   // dl
                                              mesh.d().data(),    // d
                                              mesh.du().data(),   // du
                                              mesh.dw().data(),   // dw
                                              x.data(),           // x
                                              mesh.dims().data(), // t_dims
                                              mesh.dims().size(), // t_ndims
                                              mesh.solve_dim());  // t_solvedim
  std::vector<char> buffer(extent, 0);
  // Dry run
  pentadsolver_gpsv_batch(handle,             // context
                          mesh.ds().data(),   // ds
                          mesh.dl().data(),   // dl
                          mesh.d().data(),    // d
                          mesh.du().data(),   // du
                          mesh.dw().data(),   // dw
                          x.data(),           // x
                          mesh.dims().data(), // t_dims
                          mesh.dims().size(), // t_ndims
                          mesh.solve_dim(),   // t_solvedim
                          buffer.data());     // t_buffer
  handle->total_sec    = 0.0;
  handle->forward_sec  = 0.0;
  handle->reduced_sec  = 0.0;
  handle->backward_sec = 0.0;
  // Solve the equations
  x = mesh.x();
  MPI_Barrier(MPI_COMM_WORLD);
  while (num_iters--) {
    pentadsolver_gpsv_batch(handle,             // context
                            mesh.ds().data(),   // ds
                            mesh.dl().data(),   // dl
                            mesh.d().data(),    // d
                            mesh.du().data(),   // du
                            mesh.dw().data(),   // dw
                            mesh.x().data(),    // x
                            mesh.dims().data(), // t_dims
                            mesh.dims().size(), // t_ndims
                            mesh.solve_dim(),   // t_solvedim
                            buffer.data());     // t_buffer
    x = mesh.x();
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
#else
#include "util/device_mesh.hpp"
#include "util/device_mesh.hpp"    // for DeviceMesh
#include "pentadsolver_cuda.hpp"   // for pentadsolver_gpsv_batch
#include "pentadsolver_handle.hpp" // for pentadsolver_handle_implementation
template <typename Float> using MyMesh = DeviceMesh<Float>;

template <typename Float>
void run_solver(pentadsolver_handle_t handle, MyMesh<Float> &mesh,
                int num_iters) {

  size_t workspace_size_h = 0;
  size_t workspace_size_d = 0;
  pentadsolver_gpsv_batch_buffer_size_ext(handle,               // context
                                          mesh.ds_d(),          // ds
                                          mesh.dl_d(),          // dl
                                          mesh.d_d(),           // d
                                          mesh.du_d(),          // du
                                          mesh.dw_d(),          // dw
                                          mesh.x_d(),           // x
                                          local_sizes.data(),   // t_dims
                                          mesh_h.dims().size(), // t_ndims
                                          mesh_h.solve_dim(),   // t_solvedim
                                          &workspace_size_h, &workspace_size_d);
  std::vector<char> buffer(workspace_size_h, 0);
  void *workspace_d = nullptr;
  cudaMalloc(&workspace_d, workspace_size_d);

  // Dry run
  pentadsolver_gpsv_batch(handle,               // context
                          mesh.ds_d(),          // ds
                          mesh.dl_d(),          // dl
                          mesh.d_d(),           // d
                          mesh.du_d(),          // du
                          mesh.dw_d(),          // dw
                          mesh.x_d(),           // x
                          local_sizes.data(),   // t_dims
                          mesh_h.dims().size(), // t_ndims
                          mesh_h.solve_dim(),   // t_solvedim
                          buffer.data(),        // t_buffer_h
                          workspace_d);         // t_buffer_d
  cudaMemcpy(mesh.x_d(), mesh.x().data(), sizeof(Float) * mesh.x().size(),
             cudaMemcpyHostToDevice);
  handle->total_sec    = 0.0;
  handle->forward_sec  = 0.0;
  handle->reduced_sec  = 0.0;
  handle->backward_sec = 0.0;
  // Solve the equations
  MPI_Barrier(MPI_COMM_WORLD);
  while (num_iters--) {
    pentadsolver_gpsv_batch(handle,               // context
                            mesh.ds_d(),          // ds
                            mesh.dl_d(),          // dl
                            mesh.d_d(),           // d
                            mesh.du_d(),          // du
                            mesh.dw_d(),          // dw
                            mesh.x_d(),           // x
                            local_sizes.data(),   // t_dims
                            mesh_h.dims().size(), // t_ndims
                            mesh_h.solve_dim(),   // t_solvedim
                            buffer.data(),        // t_buffer_h
                            workspace_d);         // t_buffer_d
    cudaMemcpy(mesh.x_d(), mesh.x().data(), sizeof(Float) * mesh.x().size(),
               cudaMemcpyHostToDevice);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
#endif

void print_local_sizes(int rank, int num_proc, const int *mpi_dims,
                       const std::vector<int> &mpi_coords,
                       const std::vector<int> &local_sizes) {
#ifdef NDEBUG
  if (rank == 0) {
    int i = 0;
#else
  for (int i = 0; i < num_proc; ++i) {
    // Print the outputs
    if (i == rank) {
#endif
    std::string idx    = std::to_string(mpi_coords[0]);
    std::string dims   = std::to_string(local_sizes[0]);
    std::string m_dims = std::to_string(mpi_dims[0]);
    for (size_t j = 1; j < local_sizes.size(); ++j) {
      idx += "," + std::to_string(mpi_coords[j]);
      dims += "x" + std::to_string(local_sizes[j]);
      m_dims += "x" + std::to_string(mpi_dims[j]);
    }
    if (rank == 0) {
      std::cout << "########## Local decomp sizes {" + m_dims +
                       "} ##########\n";
    }
    std::cout << "# Rank " << i << "(" + idx + "){" + dims + "}\n";
#ifdef NDEBUG
  }
#else
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
}

void usage(const char *name) {
  std::cerr << "Usage:\n";
  std::cerr
      << "\t" << name
      << " [-x nx -y ny -z nz -d ndims -s solvedim -b batch_size -m "
         "mpir_strat_idx -p num_partitions_along_solvedim] -n num_iterations"
      << std::endl;
}

void print_header(int rank, const char *executable, int ndims, int solvedim,
                  int num_proc, const std::vector<int> &dims) {
  if (rank == 0) {
    std::string fname = executable;
    fname             = fname.substr(fname.rfind("/") + 1);
    std::cout << fname << " " << ndims << "DS" << solvedim << "NP" << num_proc;
    std::cout << " {" << dims[0];
    for (size_t i = 1; i < dims.size(); ++i)
      std::cout << "x" << dims[i];
    std::cout << "}";
    std::cout << " solvedim" << solvedim << "\n";
  }
}

void report_single(std::string name, double val) {
  int rank  = 0;
  int nproc = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  std::vector<double> times(nproc, 0);
  MPI_Gather(&val, 1, MPI_DOUBLE, times.data(), 1, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);
  if (rank == 0) {
    double mean = 0.0;
    double max  = times[0];
    double min  = times[0];
    for (double t : times) {
      mean += t;
      max = std::max(max, t);
      min = std::min(min, t);
    }
    mean = mean / nproc;
    double stddev =
        std::accumulate(times.begin(), times.end(), 0.0,
                        [&](const double &sum, const double &time) {
                          return sum + (time - mean) * (time - mean);
                        });
    stddev = std::sqrt(stddev / nproc);

    std::cout << name << ": ";
    std::cout << min << "s; " << max << "s; " << mean << "s; " << stddev
              << "s;\n";
  }
}
void profile_report(pentadsolver_handle_t handle) {
  report_single("total", handle->total_sec);
  report_single("forward", handle->forward_sec);
  report_single("reduced", handle->reduced_sec);
  report_single("backward", handle->backward_sec);
}

template <typename Float>
void test_solver_with_generated(const char *execname,
                                const std::vector<int> &global_dims,
                                int solvedim, int mpi_parts_in_s,
                                std::array<int, 3> mpi_dims_, int num_iters) {
  int num_proc = 0;
  int rank     = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(mpi_dims_.data(),
                            mpi_dims_.data() + global_dims.size());
  std::vector<int> periods(global_dims.size(), 0);
  mpi_dims[solvedim] = std::min(num_proc, mpi_parts_in_s);
  MPI_Dims_create(num_proc, static_cast<int>(global_dims.size()),
                  mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm = nullptr;
  MPI_Cart_create(MPI_COMM_WORLD, static_cast<int>(global_dims.size()),
                  mpi_dims.data(), periods.data(), 0, &cart_comm);

  pentadsolver_handle_t handle{};
  pentadsolver_create(&handle, &cart_comm, static_cast<int>(global_dims.size()),
                      mpi_dims.data());

  // The size of the local domain.
  std::vector<int> local_sizes(global_dims.size());
  // The starting indices of the local domain in each dimension.
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = global_dims[i];
    size_t domain_offset = handle->mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i]       = handle->mpi_coords[i] == mpi_dims[i] - 1
                               ? global_dim - domain_offset
                               : global_dim / mpi_dims[i];
  }

  print_local_sizes(rank, num_proc, handle->num_mpi_procs, handle->mpi_coords,
                    local_sizes);

  MyMesh<Float> mesh(solvedim, local_sizes);
  print_header(rank, execname, global_dims.size(), solvedim, num_proc,
               global_dims);
  MPI_Barrier(MPI_COMM_WORLD);
  run_solver(handle, mesh, num_iters);
  profile_report(handle);
}

int main(int argc, char *argv[]) {
  auto rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  int num_proc = 0;
  int rank     = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  int opt                     = 0;
  std::array<int, 3> size     = {256, 256, 256};
  int ndims                   = 2;
  int solvedim                = 0;
  int num_iters               = 1;
  int mpi_parts_in_s          = 0; // 0 means automatic
  std::array<int, 3> mpi_dims = {0, 0, 0};
  while ((opt = getopt(argc, argv, "x:y:z:s:d:p:n:X:Y:Z:")) != -1) {
    switch (opt) {
    case 'x': size[0] = atoi(optarg); break;
    case 'y': size[1] = atoi(optarg); break;
    case 'z': size[2] = atoi(optarg); break;
    case 'd': ndims = atoi(optarg); break;
    case 's': solvedim = atoi(optarg); break;
    case 'n': num_iters = atoi(optarg); break;
    case 'p': mpi_parts_in_s = atoi(optarg); break;
    case 'X': mpi_dims[0] = atoi(optarg); break;
    case 'Y': mpi_dims[1] = atoi(optarg); break;
    case 'Z': mpi_dims[2] = atoi(optarg); break;
    default:
      if (rank == 0) usage(argv[0]);
      return 2;
      break;
    }
  }

  std::vector<int> dims;
  for (int i = 0; i < ndims; ++i) {
    dims.push_back(size[i]);
  }

  test_solver_with_generated<double>(argv[0], dims, solvedim, mpi_parts_in_s,
                                     mpi_dims, num_iters);

  MPI_Finalize();
  return 0;
}
