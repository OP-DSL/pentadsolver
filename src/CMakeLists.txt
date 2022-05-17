option(BUILD_FOR_CPU "Build library for CPU architecture." ON)

if(BUILD_FOR_CPU)
  add_subdirectory(cpu)
endif(BUILD_FOR_CPU)
