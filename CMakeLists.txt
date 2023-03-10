cmake_minimum_required(VERSION 3.0.2)
project(orb_slam3)
add_definitions(-w)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    add_definitions(-DCOMPILEDWITHC14)
    message(STATUS "Using flag -std=c++14.")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif ()

# set(CMAKE_BUILD_TYPE Debug)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(catkin REQUIRED COMPONENTS
	cv_bridge
	image_transport
	roscpp
	sensor_msgs
	tf
)

find_package(OpenCV 4.2.0 QUIET)
if (NOT OpenCV_FOUND)
	find_package(OpenCV 3.2 QUIET)
	if (NOT OpenCV_FOUND)
		message(FATAL_ERROR "OpenCV > 3.2 not found.")
	endif ()
endif ()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

catkin_package()

include_directories(
	${catkin_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/include/CameraModels
	${PROJECT_SOURCE_DIR}/include/moCamera/mo_stereo_camera
	${PROJECT_SOURCE_DIR}/include/moCamera/mo_stereo_camera_reserved
	${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
	${EIGEN3_INCLUDE_DIR}
	${Pangolin_INCLUDE_DIRS}
)


link_directories(
  	${PROJECT_SOURCE_DIR}/lib/moCamera
)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(metoak_libs
	moStereoCameraDriver
	pthread
	${OpenCV_LIBS}
	)

add_library(${PROJECT_NAME} SHARED
	src/System.cc
	src/Tracking.cc
	src/LocalMapping.cc
	src/LoopClosing.cc
	src/ORBextractor.cc
	src/ORBmatcher.cc
	src/FrameDrawer.cc
	src/Converter.cc
	src/MapPoint.cc
	src/KeyFrame.cc
	src/Atlas.cc
	src/Map.cc
	src/MapDrawer.cc
	src/Optimizer.cc
	src/Frame.cc
	src/KeyFrameDatabase.cc
	src/Sim3Solver.cc
	src/Viewer.cc
	src/ImuTypes.cc
	src/G2oTypes.cc
	src/CameraModels/Pinhole.cpp
	src/CameraModels/KannalaBrandt8.cpp
	src/OptimizableTypes.cpp
	src/MLPnPsolver.cpp
	src/GeometricTools.cc
	src/TwoViewReconstruction.cc
	src/Config.cc
	src/Settings.cc
	include/System.h
	include/Tracking.h
	include/LocalMapping.h
	include/LoopClosing.h
	include/ORBextractor.h
	include/ORBmatcher.h
	include/FrameDrawer.h
	include/Converter.h
	include/MapPoint.h
	include/KeyFrame.h
	include/Atlas.h
	include/Map.h
	include/MapDrawer.h
	include/Optimizer.h
	include/Frame.h
	include/KeyFrameDatabase.h
	include/Sim3Solver.h
	include/Viewer.h
	include/ImuTypes.h
	include/G2oTypes.h
	include/CameraModels/GeometricCamera.h
	include/CameraModels/Pinhole.h
	include/CameraModels/KannalaBrandt8.h
	include/OptimizableTypes.h
	include/MLPnPsolver.h
	include/GeometricTools.h
	include/TwoViewReconstruction.h
	include/SerializationUtils.h
	include/Config.h
	include/Settings.h)

add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/g2o)
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
	${EIGEN3_LIBS}
	${Pangolin_LIBRARIES}
	DBoW2
	g2o
	boost_system
	boost_serialization
	crypto
	)

set(ros_libs
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
	${EIGEN3_LIBS}
	${Pangolin_LIBRARIES}
	DBoW2
	g2o
	orb_slam3
	boost_system
	)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/METOAK)
add_executable(mo_rgbd METOAK/mo_rgbd.cpp METOAK/common_function.cpp)
target_link_libraries(mo_rgbd ${metoak_libs} ${ros_libs})

add_executable(mo_rgbd_raw METOAK/mo_rgbd_raw.cpp METOAK/common_function.cpp)
target_link_libraries(mo_rgbd_raw ${metoak_libs} ${PROJECT_NAME})

add_executable(mo_stereo METOAK/mo_stereo.cpp METOAK/common_function.cpp)
target_link_libraries(mo_stereo ${metoak_libs} ${ros_libs})

add_executable(ros_rgbd METOAK/ros_rgbd.cpp)
target_link_libraries(ros_rgbd ${ros_libs})

add_executable(ros_rgbd_imu METOAK/ros_rgbd_imu.cpp)
target_link_libraries(ros_rgbd_imu ${ros_libs})

add_executable(ros_stereo METOAK/ros_stereo.cpp)
target_link_libraries(ros_stereo ${ros_libs})
