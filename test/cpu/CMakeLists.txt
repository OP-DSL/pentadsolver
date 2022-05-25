add_executable(tests tests.cpp)
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
