cmake_minimum_required(VERSION 2.8.3)
project(yak_msgs)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

add_service_files(
  DIRECTORY srv
  FILES
    ResetDimensions.srv
    ResetParams.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
)
