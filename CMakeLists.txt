cmake_minimum_required(VERSION 2.8)
project(nn-optim)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall")
elseif (MSVC)
	
endif()


set(SOURCES 
	source/main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

add_subdirectory(libnn)
include_directories(libnn/include)
include_directories(libnn/libla)

if(OPENCL)
	include_directories(libnn/libcl4u/include)
endif(OPENCL)

target_link_libraries(${PROJECT_NAME} nn)
