# For use in subdirs.
set(SPARSEKERNEL_SRCS)

set(SPARSEKERNEL_SPMM_TEST_SRCS)
set(SPARSEKERNEL_SPMM_BENCHMARK_SRCS)
add_subdirectory(spmm)

set(SPARSEKERNEL_SDDMM_TEST_SRCS)
set(SPARSEKERNEL_SDDMM_BENCHMARK_SRCS)
add_subdirectory(sddmm)

set(SPARSEKERNEL_DEPTHWISE_TEST_SRCS)
set(SPARSEKERNEL_DEPTHWISE_BENCHMARK_SRCS)
add_subdirectory(depthwise)

add_subdirectory(bias_relu)
add_subdirectory(softmax)
add_subdirectory(utils)

set(DIR_SRCS)

##
### Find all sources in this directory.
##

# Lib srcs.
file(GLOB TMP *.h)
list(APPEND DIR_SRCS ${TMP})
file(GLOB TMP *.cc)
list(APPEND DIR_SRCS ${TMP})

##
### Filter files that we don't want in the main library.
##

set(FILTER_SRCS)

# Don't want test related code.
file(GLOB TMP test_utils*)
list(APPEND FILTER_SRCS ${TMP})

# Don't need matrix utilities.
file(GLOB TMP matrix_utils*)
list(APPEND FILTER_SRCS ${TMP})

foreach(FILE ${FILTER_SRCS})
  list(REMOVE_ITEM DIR_SRCS ${FILE})
endforeach(FILE)

# Add the sources to the build.
list(APPEND SPARSEKERNEL_SRCS ${DIR_SRCS})

# Set .cu.cc files to be compiled as CUDA.
set(SPARSEKERNEL_CUDA_SRCS ${SPARSEKERNEL_SRCS})
list(FILTER SPARSEKERNEL_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
set_source_files_properties(${SPARSEKERNEL_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

# Create libsparsekernel.
add_library(sparsekernel SHARED ${SPARSEKERNEL_SRCS})
target_link_libraries(sparsekernel ${SPARSEKERNEL_LIBS})

# Library installation.
install(TARGETS sparsekernel)

set(INSTALL_BASE "include/sparsekernel")
install(FILES "cuda_utils.h" DESTINATION ${INSTALL_BASE})
install(FILES "sparsekernel.h" DESTINATION ${INSTALL_BASE})
install(FILES "bias_relu/bias_relu.h" DESTINATION "${INSTALL_BASE}/bias_relu")
install(FILES "depthwise/cuda_depthwise.h" DESTINATION "${INSTALL_BASE}/depthwise")
install(FILES "spmm/cuda_spmm.h" DESTINATION "${INSTALL_BASE}/spmm")
install(FILES "sddmm/cuda_sddmm.h" DESTINATION "${INSTALL_BASE}/sddmm")
install(FILES "softmax/softmax.h" DESTINATION "${INSTALL_BASE}/softmax")
install(FILES "softmax/sparse_softmax.h" DESTINATION "${INSTALL_BASE}/softmax")
install(FILES "utils/index_format.h" DESTINATION "${INSTALL_BASE}/utils")

# Optionally build the test suite.
if (BUILD_TEST)
  # Test sources for all targets.
  set(SPARSEKERNEL_TEST_SRCS)
  file(GLOB TMP matrix_utils*)
  list(APPEND SPARSEKERNEL_TEST_SRCS ${TMP})
  file(GLOB TMP test_utils*)
  list(APPEND SPARSEKERNEL_TEST_SRCS ${TMP})

  # SpMM test build.
  list(APPEND SPARSEKERNEL_SPMM_TEST_SRCS ${SPARSEKERNEL_TEST_SRCS})
  
  set(SPARSEKERNEL_SPMM_TEST_CUDA_SRCS ${SPARSEKERNEL_SPMM_TEST_SRCS})
  list(FILTER SPARSEKERNEL_SPMM_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPARSEKERNEL_SPMM_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(spmm_test ${SPARSEKERNEL_SPMM_TEST_SRCS})
  target_link_libraries(spmm_test sparsekernel ${SPARSEKERNEL_TEST_LIBS})

  # SDDMM test build.
  list(APPEND SPARSEKERNEL_SDDMM_TEST_SRCS ${SPARSEKERNEL_TEST_SRCS})
  
  set(SPARSEKERNEL_SDDMM_TEST_CUDA_SRCS ${SPARSEKERNEL_SDDMM_TEST_SRCS})
  list(FILTER SPARSEKERNEL_SDDMM_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPARSEKERNEL_SDDMM_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(sddmm_test ${SPARSEKERNEL_SDDMM_TEST_SRCS})
  target_link_libraries(sddmm_test sparsekernel ${SPARSEKERNEL_TEST_LIBS})

  # Depthwise test build.
  list(APPEND SPARSEKERNEL_DEPTHWISE_TEST_SRCS ${SPARSEKERNEL_TEST_SRCS})
  
  set(SPARSEKERNEL_DEPTHWISE_TEST_CUDA_SRCS ${SPARSEKERNEL_DEPTHWISE_TEST_SRCS})
  list(FILTER SPARSEKERNEL_DEPTHWISE_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(
    ${SPARSEKERNEL_DEPTHWISE_TEST_CUDA_SRCS}
    PROPERTIES LANGUAGE CUDA)  
  add_executable(depthwise_test ${SPARSEKERNEL_DEPTHWISE_TEST_SRCS})
  target_link_libraries(depthwise_test sparsekernel ${SPARSEKERNEL_TEST_LIBS})
endif()

# Optionally build the benchmark suite.
if (BUILD_BENCHMARK)
  # Benchmark sources for all targets.
  set(SPARSEKERNEL_BENCHMARK_SRCS)
  file(GLOB TMP matrix_utils*)
  list(APPEND SPARSEKERNEL_BENCHMARK_SRCS ${TMP})
  file(GLOB TMP test_utils*)
  list(APPEND SPARSEKERNEL_BENCHMARK_SRCS ${TMP})
  
  # SpMM benchmark build.
  list(APPEND SPARSEKERNEL_SPMM_BENCHMARK_SRCS ${SPARSEKERNEL_BENCHMARK_SRCS})
  
  set(SPARSEKERNEL_SPMM_BENCHMARK_CUDA_SRCS ${SPARSEKERNEL_SPMM_BENCHMARK_SRCS})
  list(FILTER SPARSEKERNEL_SPMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPARSEKERNEL_SPMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(spmm_benchmark ${SPARSEKERNEL_SPMM_BENCHMARK_SRCS})
  target_link_libraries(spmm_benchmark sparsekernel ${SPARSEKERNEL_BENCHMARK_LIBS})

  # SDDMM benchmark build.
  list(APPEND SPARSEKERNEL_SDDMM_BENCHMARK_SRCS ${SPARSEKERNEL_BENCHMARK_SRCS})
  
  set(SPARSEKERNEL_SDDMM_BENCHMARK_CUDA_SRCS ${SPARSEKERNEL_SDDMM_BENCHMARK_SRCS})
  list(FILTER SPARSEKERNEL_SDDMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPARSEKERNEL_SDDMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(sddmm_benchmark ${SPARSEKERNEL_SDDMM_BENCHMARK_SRCS})
  target_link_libraries(sddmm_benchmark sparsekernel ${SPARSEKERNEL_BENCHMARK_LIBS})

  # Depthwise benchmark build.
  list(APPEND SPARSEKERNEL_DEPTHWISE_BENCHMARK_SRCS ${SPARSEKERNEL_BENCHMARK_SRCS})
  
  set(SPARSEKERNEL_DEPTHWISE_BENCHMARK_CUDA_SRCS ${SPARSEKERNEL_DEPTHWISE_BENCHMARK_SRCS})
  list(FILTER SPARSEKERNEL_DEPTHWISE_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(
    ${SPARSEKERNEL_DEPTHWISE_BENCHMARK_CUDA_SRCS}
    PROPERTIES LANGUAGE CUDA)  
  add_executable(depthwise_benchmark ${SPARSEKERNEL_DEPTHWISE_BENCHMARK_SRCS})
  target_link_libraries(depthwise_benchmark sparsekernel ${SPARSEKERNEL_BENCHMARK_LIBS})    
endif()
