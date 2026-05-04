cmake_minimum_required(VERSION 3.20)
project(synth_cpp_port LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MINGW)
    add_link_options(-static-libstdc++ -static-libgcc)
endif()

add_library(synth_core
    src/core.cpp
    src/benchmarks.cpp
    src/current_oracle.cpp
)

target_include_directories(synth_core PUBLIC include)

if(MSVC)
    target_compile_options(synth_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(synth_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(synth_search apps/synth_search.cpp)
target_link_libraries(synth_search PRIVATE synth_core)

add_executable(synth_verify apps/synth_verify.cpp)
target_link_libraries(synth_verify PRIVATE synth_core)

add_executable(synth_codegen apps/synth_codegen.cpp)
target_link_libraries(synth_codegen PRIVATE synth_core)

add_executable(synth_benchmark apps/synth_benchmark.cpp)
target_link_libraries(synth_benchmark PRIVATE synth_core)

add_executable(synth_demo_build apps/synth_demo_build.cpp)
target_link_libraries(synth_demo_build PRIVATE synth_core)

add_executable(synth_tests tests/synth_tests.cpp)
target_link_libraries(synth_tests PRIVATE synth_core)

enable_testing()
add_test(NAME synth_tests COMMAND synth_tests)
