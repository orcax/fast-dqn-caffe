cmake_minimum_required (VERSION 2.8.9)
project(fast_dqn)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -Wno-deprecated-declarations")

# cuda
set(CUDA_DIR /usr/local/cuda)
link_directories(${CUDA_DIR}/lib64)
include_directories(${CUDA_DIR}/include)

# caffe
set(CAFFE_LIB libcaffe.so)
link_directories(./caffe/build/lib)
include_directories(./caffe/include)

# ale
set(ALE_LIB libale.so)
link_directories(./ale)
include_directories(./ale/src)

# opencv
set(OPENCV_LIB libopencv_core.so libopencv_imgproc.so libopencv_highgui.so)

add_executable(fast_dqn 
  src/fast_dqn_main.cpp 
  src/fast_dqn.cpp 
  src/ale_environment.cpp)
target_link_libraries(fast_dqn ${ALE_LIB} ${CAFFE_LIB} ${OPENCV_LIB})

add_executable(data_gen
  src/data_generator.cpp
  src/fast_dqn.cpp
  src/ale_environment.cpp)
target_link_libraries(data_gen ${ALE_LIB} ${CAFFE_LIB} ${OPENCV_LIB})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

find_package(Boost 1.40 COMPONENTS filesystem system iostreams regex REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(fast_dqn ${Boost_LIBRARIES})
target_link_libraries(data_gen ${Boost_LIBRARIES})

find_package(GFLAGS REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})
target_link_libraries(fast_dqn ${GFLAGS_LIBRARY})
target_link_libraries(data_gen ${GFLAGS_LIBRARY})

find_package(GLOG REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
target_link_libraries(fast_dqn ${GLOG_LIBRARIES})
target_link_libraries(data_gen ${GLOG_LIBRARIES})

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
target_link_libraries(fast_dqn ${PROTOBUF_LIBRARIES})
target_link_libraries(data_gen ${PROTOBUF_LIBRARIES})

find_package(HDF5 COMPONENTS HL QUIET)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
list(APPEND LINK_LIBS ${HDF5_LIBRARIES})

