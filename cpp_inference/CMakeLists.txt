cmake_minimum_required(VERSION 3.12)
project(its_cpp_inference LANGUAGES CXX)

# ---- Chuan C++ va kieu build mac dinh ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # Release (-O3) la bat buoc de do toc do co y nghia
endif()

# ---- OpenCV (>= 4.5, can module dnn, imgproc, videoio, highgui) ----
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc dnn videoio highgui)
message(STATUS "OpenCV ${OpenCV_VERSION} tai ${OpenCV_DIR}")

# ---- Thu vien loi: Detector + AlertManager + doc cau hinh ----
add_library(its_core STATIC
  src/its_config.cpp
  src/detector.cpp
  src/alert.cpp)
target_include_directories(its_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(its_core PUBLIC ${OpenCV_LIBS})
if(MSVC)
  target_compile_options(its_core PUBLIC /W3)
else()
  target_compile_options(its_core PUBLIC -Wall -Wextra -O3)
endif()

# ---- Ung dung chinh va chuong trinh do hieu nang ----
add_executable(its_detector src/main.cpp)
target_link_libraries(its_detector PRIVATE its_core)

add_executable(its_benchmark src/benchmark.cpp)
target_link_libraries(its_benchmark PRIVATE its_core)

# Dat file thuc thi vao cpp_inference/build/bin
set_target_properties(its_detector its_benchmark PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
