cmake_minimum_required(VERSION 3.20)
project(reflex-infer VERSION 0.1.0 LANGUAGES CXX)

add_library(reflex_infer STATIC
    src/dispatch.cpp
)
add_library(reflex::infer ALIAS reflex_infer)

option(REFLEX_INFER_ENABLE_CUDA
    "Build Jetson CUDA kernels when a CUDA compiler is available"
    ON)
set(REFLEX_INFER_CUDA_ARCHITECTURES "87" CACHE STRING
    "CUDA architectures for reflex-infer kernels")

target_compile_features(reflex_infer PUBLIC cxx_std_17)
target_include_directories(reflex_infer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(reflex_infer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(reflex_infer PUBLIC
    REFLEX_INFER_VERSION="${PROJECT_VERSION}"
)

set_target_properties(reflex_infer PROPERTIES
    EXPORT_NAME infer
    POSITION_INDEPENDENT_CODE ON
)

include(CheckLanguage)
check_language(CUDA)
if(REFLEX_INFER_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set(REFLEX_INFER_CONFIG_HAS_CUDA TRUE)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(reflex_infer PRIVATE
        src/kernels/q4_gemm.cu
        src/kernels/attention.cu
    )
    target_link_libraries(reflex_infer PUBLIC CUDA::cudart)
    target_compile_definitions(reflex_infer PUBLIC REFLEX_INFER_HAS_CUDA=1)
    set_target_properties(reflex_infer PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${REFLEX_INFER_CUDA_ARCHITECTURES}"
    )
    target_compile_options(reflex_infer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>
    )
else()
    set(REFLEX_INFER_CONFIG_HAS_CUDA FALSE)
    target_sources(reflex_infer PRIVATE
        src/kernels/cuda_stubs.cpp
    )
    target_compile_definitions(reflex_infer PUBLIC REFLEX_INFER_HAS_CUDA=0)
endif()

include(CMakePackageConfigHelpers)

install(TARGETS reflex_infer
    EXPORT reflex-inferTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT reflex-inferTargets
    NAMESPACE reflex::
    DESTINATION lib/cmake/reflex-infer
)

configure_package_config_file(
    cmake/reflex-inferConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/reflex-inferConfig.cmake"
    INSTALL_DESTINATION lib/cmake/reflex-infer
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/reflex-inferConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/reflex-inferConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/reflex-inferConfigVersion.cmake"
    DESTINATION lib/cmake/reflex-infer
)
