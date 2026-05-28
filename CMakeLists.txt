cmake_minimum_required(VERSION 3.20)
project(reflex-infer VERSION 0.1.0 LANGUAGES CXX)

add_library(reflex_infer INTERFACE)
add_library(reflex::infer ALIAS reflex_infer)

target_compile_features(reflex_infer INTERFACE cxx_std_17)
target_include_directories(reflex_infer INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(reflex_infer INTERFACE
    REFLEX_INFER_VERSION="${PROJECT_VERSION}"
)

set_target_properties(reflex_infer PROPERTIES EXPORT_NAME infer)

include(CMakePackageConfigHelpers)

install(TARGETS reflex_infer
    EXPORT reflex-inferTargets
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

