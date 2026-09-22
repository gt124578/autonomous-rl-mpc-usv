cmake_minimum_required(VERSION 3.0.2)
project(myboat_gazebo)

find_package(catkin REQUIRED COMPONENTS gazebo_dev roscpp message_generation xacro wave_gazebo_plugins usv_msgs)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS message_runtime gazebo_dev roscpp wave_gazebo_plugins
)

# Plugins require c++11
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

include_directories( include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
link_directories(
  ${catkin_LIBRARY_DIRS}
)

# 推进器
add_library(myboat_thrust_plugin
  src/myboat_thrust_plugin.cpp
)
target_link_libraries(myboat_thrust_plugin
  ${catkin_LIBRARIES}
  ${Eigen_LIBRARIES}
)
install(TARGETS myboat_thrust_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 动力学
add_library(myboat_dynamics_plugin
  src/myboat_dynamics_plugin.cpp
)
target_link_libraries(myboat_dynamics_plugin
  ${catkin_LIBRARIES}
  ${Eigen_LIBRARIES}
)
install(TARGETS myboat_dynamics_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)


install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

# test cpp
add_executable(test_cpp_myboat_control src/test_cpp_myboat_control.cpp)
target_link_libraries(test_cpp_myboat_control
  ${catkin_LIBRARIES}
)

# # 设置依赖（可选但推荐）
# add_dependencies(myboat_control_node
#   ${${PROJECT_NAME}_EXPORTED_TARGETS}
#   ${catkin_EXPORTED_TARGETS}
# )