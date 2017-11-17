project(aMAZEd)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
cmake_minimum_required(VERSION 3.8)

# Where to find CMake modules and OpenCV
set(OpenCV_DIR "C:\\opencv\\opencv-3.2.0\\build")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(aMAZEd main.cpp stream/CameraStream.cpp stream/CameraStream.h analyse/EdgeDetection.cpp analyse/EdgeDetection.h
        modelisation/Transformation.cpp modelisation/Transformation.h modelisation/OpenGL.cpp modelisation/OpenGL.h
        modelisation/GlutMaster.cpp modelisation/GlutMaster.h modelisation/GlutWindow.cpp modelisation/GlutWindow.h
        physics/Ball.cpp physics/Ball.h physics/AngleModel.cpp physics/AngleModel.h physics/CollisionDetection.cpp
        physics/CollisionDetection.h physics/Vector2d.cpp physics/Vector2d.h physics/CollisionDetection.cpp physics/CollisionDetection.h
        physics/Vector2d.cpp physics/Vector2d.h physics/Wall.cpp physics/Wall.h)

# add OpenCV libs
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_videoio opencv_calib3d)

# add OpenGL libs
set(OpenGL_LIBS libopengl32.a libglu32.a libfreeglut.a)

# linking
target_link_libraries(aMAZEd ${OpenCV_LIBS} ${OpenGL_LIBS})
