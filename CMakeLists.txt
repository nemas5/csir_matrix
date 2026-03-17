cmake_minimum_required(VERSION 3.10)
project(csir)

set(CMAKE_CXX_STANDARD 17)

add_executable(csir
    main.cpp
    CSIRMatrix/CSIRMatrix.cpp
)

target_include_directories(csir PRIVATE
    ${CMAKE_SOURCE_DIR}
)

find_package(OpenMP REQUIRED)
target_link_libraries(csir PRIVATE OpenMP::OpenMP_CXX)