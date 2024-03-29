add_executable(tests singlenode/tests.cpp)
target_link_libraries(
  tests
  PRIVATE pentadsolver::project_warnings
          pentadsolver::project_options
          pentadsolver::cpu
          catch_main)

target_include_directories(tests PUBLIC ${PROJECT_SOURCE_DIR})

# automatically discover tests that are defined in catch based test files you can modify the unittests. Set TEST_PREFIX
# to whatever you want, or use different for different binaries
catch_discover_tests(
  tests
  TEST_PREFIX
  "unittests."
  REPORTER
  xml
  WORKING_DIRECTORY
  ${PROJECT_SOURCE_DIR}
  OUTPUT_DIR
  .OUTPUT_PREFIX
  "unittests."
  OUTPUT_SUFFIX.xml)

if(BUILD_FOR_MPI)
  find_package(MPI REQUIRED)

  add_executable(mpi_tests mpi/mpi_main.cpp mpi/tests.cpp)
  target_link_libraries(
    mpi_tests
    PRIVATE pentadsolver::project_warnings
            pentadsolver::project_options
            pentadsolver::mpi_cpu
            Catch2::Catch2)

  target_include_directories(mpi_tests PUBLIC ${PROJECT_SOURCE_DIR})

  catch_discover_tests(
    mpi_tests
    TEST_PREFIX
    "unittests."
    REPORTER
    xml
    WORKING_DIRECTORY
    ${PROJECT_SOURCE_DIR}
    OUTPUT_DIR
    .OUTPUT_PREFIX
    "unittests."
    OUTPUT_SUFFIX.xml)

endif(BUILD_FOR_MPI)
