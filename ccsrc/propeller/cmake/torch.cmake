# INCLUDE(ccsrc/propeller/cmake/generated/torch.cmake)

FUNCTION(build_torch_quiver_ext target)
    ADD_LIBRARY(${target} SHARED)
    # ADD_LIBRARY(${target} STATIC)
    TARGET_INCLUDE_DIRECTORIES(${target}
                               PRIVATE ${CMAKE_SOURCE_DIR}/ccsrc/propeller/include)
    TARGET_SOURCE_DIR(${target}
                          ${CMAKE_SOURCE_DIR}/ccsrc/propeller/*.cu)
    TARGET_SET_CUDA_OPTIONS(${target})
ENDFUNCTION()

BUILD_TORCH_QUIVER_EXT(propeller)