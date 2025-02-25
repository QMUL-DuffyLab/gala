cmake_minimum_required(VERSION 3.27)
enable_language(Fortran)
project(antenna)
# add_executable(libantenna antenna.f nnls.f)
include(GNUInstallDirs)
find_package(LAPACK)
find_package(BLAS)

include(FetchContent)
FetchContent_Declare(
  jsonfortran
  GIT_REPOSITORY https://github.com/jacobwilliams/json-fortran
  GIT_TAG 8.4.0
  FIND_PACKAGE_ARGS NAMES jsonfortran
)
# stop json-fortran building its docs - don't need them
set(SKIP_DOC_GEN TRUE CACHE INTERNAL "")
# hopefully stop jsonfortran being installed system wide?
set(USE_GNU_INSTALL_CONVENTION FALSE CACHE INTERNAL "")
FetchContent_MakeAvailable(jsonfortran)

set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR})

# testing stuff
message(NOTE, "cmake install libdir = ${CMAKE_INSTALL_LIBDIR}")
message(NOTE, "proj_bin_dir  = ${PROJECT_BINARY_DIR}")
message(NOTE, "proj_src_dir  = ${PROJECT_SOURCE_DIR}")
message(NOTE, "fortran compiler  = ${CMAKE_Fortran_COMPILER_ID}")
message(NOTE, "prefix  = ${CMAKE_INSTALL_PREFIX}")

set (SRCS antenna.f90 nnls.f90)
set(CMAKE_Fortran_FLAGS "-g -ggdb3 -std=f2018 -ffree-form -Wall -Werror -pedantic -fcheck=all")
set(CMAKE_EXE_LINKER_FLAGS "-llapack -lblas")
add_library(${PROJECT_NAME} SHARED ${SRCS})
target_link_libraries(${PROJECT_NAME} jsonfortran lapack blas)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  $<TARGET_PROPERTY:jsonfortran,Fortran_MODULE_DIRECTORY>
  # $<BUILD_INTERFACE:${MODULE_DIR}>
  # $<INSTALL_INTERFACE:${INSTALL_DIR}>
)
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
  OUTPUT_NAME ${PROJECT_NAME}
  PREFIX lib
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
  Fortran_MODULE_DIRECTORY ${PROJECT_SOURCE_DIR}/mod
)

install(
  TARGETS 
  ${PROJECT_NAME}
  # EXPORT ${PACKAGE_NAME}-targets
  LIBRARY DESTINATION ${PROJECT_BINARY_DIR}/lib
  ARCHIVE DESTINATION ${PROJECT_BINARY_DIR}/mod
)
