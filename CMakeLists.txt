cmake_minimum_required(VERSION 2.6)
project(urlbar-detector)

file(GLOB_RECURSE Sources src/*.cpp)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} ${Sources})
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS})
