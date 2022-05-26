cmake_minimum_required(VERSION 3.18...3.23)

project(PentadSolverPackageTests LANGUAGES CXX)

# ---- Test as standalone project the exported config package ----
# This variable is set by project() in CMake 3.21+
string(
  COMPARE EQUAL
          "${CMAKE_SOURCE_DIR}"
          "${PROJECT_SOURCE_DIR}"
          PROJECT_IS_TOP_LEVEL)
if(PROJECT_IS_TOP_LEVEL OR TEST_INSTALLED_VERSION)
  enable_testing()

  find_package(pentadsolver CONFIG REQUIRED) # for project_options, ...

  if(NOT TARGET pentadsolver::project_options)
    message(FATAL_ERROR "Requiered config package not found!")
    return() # be strictly paranoid for Template Janitor github action! CK
  endif()
endif()

# ---- Dependencies ----

find_package(Catch2 CONFIG REQUIRED)

include(Catch)

add_library(catch_main OBJECT catch_main.cpp)
target_link_libraries(catch_main PUBLIC Catch2::Catch2)
target_link_libraries(catch_main PRIVATE pentadsolver::project_options)

if(BUILD_FOR_CPU)
  add_subdirectory(cpu)
endif(BUILD_FOR_CPU)

if(BUILD_FOR_CUDA)
  add_subdirectory(cuda)
endif(BUILD_FOR_CUDA)
