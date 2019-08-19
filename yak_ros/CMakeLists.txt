cmake_minimum_required(VERSION 3.5.0)
project(yak_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(yak REQUIRED)
find_package(yak_msgs REQUIRED)

find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(cv_bridge REQUIRED)


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui)

find_package(OpenMP REQUIRED)

if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
    set_property(TARGET OpenMP::OpenMP_CXX
      PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif()

# trying to avoid setting LD_LIBRARY_PATH
set(NVIDIA_LIBS /usr/lib/nvidia-384)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "/usr/lib/nvidia-384")

if ($ENV{ROS_VERSION} VERSION_EQUAL "1")
    # Build for ROS 1 using Catkin
    find_package(gl_depth_sim REQUIRED)

     find_package(catkin REQUIRED COMPONENTS
         roscpp
         tf2
         tf2_eigen
         cv_bridge)

     catkin_package(
       INCLUDE_DIRS
       LIBRARIES
         ${OpenCV_LIBS}
         ${yak_LIBRARIES}
         ${sensor_msgs_LIBRARIES}
         ${tf2_LIBRARIES}
         ${tf2_ros_LIBRARIES}
         ${tf2_eigen_LIBRARIES}
         ${cv_bridge_LIBRARIES}
	 ${NVIDIA_LIBS}
       CATKIN_DEPENDS
       DEPENDS
         yak
	 yak_msgs
         tf2_eigen
         tf2
     )

     add_executable(${PROJECT_NAME}_node
         src/yak_ros1_node.cpp)
     target_link_libraries(${PROJECT_NAME}_node
         ${CATKIN_LIBRARIES}
         ${OpenCV_LIBS}
         ${yak_LIBRARIES}
         ${sensor_msgs_LIBRARIES}
         ${tf2_LIBRARIES}
         ${tf2_ros_LIBRARIES}
         ${tf2_eigen_LIBRARIES}
         ${cv_bridge_LIBRARIES}
	 ${NVIDIA_LIBS}
         )

     target_include_directories(${PROJECT_NAME}_node PUBLIC
       ${catkin_INCLUDE_DIRS}
       ${yak_INCLUDE_DIRS}
       ${yak_msgs_INCLUDE_DIRS}
       ${gl_depth_sim_INCLUDE_DIRS}
     )
   include_directories(include)
   
else()
    # Build for ROS 2 using Ament
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)

    add_executable(${PROJECT_NAME}_node
        src/yak_ros2_node.cpp)

    target_include_directories(${PROJECT_NAME}_node PUBLIC
      ${rclcpp_INCLUDE_DIRS}
      ${yak_INCLUDE_DIRS}
    )

    target_link_libraries(${PROJECT_NAME}_node
        ${rclcpp_LIBRARIES}
        ${OpenCV_LIBS}
        ${yak_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${tf2_LIBRARIES}
        ${tf2_ros_LIBRARIES}
        ${tf2_eigen_LIBRARIES}
        ${cv_bridge_LIBRARIES}
    )

    ament_target_dependencies(${PROJECT_NAME}_node
                              "rclcpp"
                              "yak"
                              "std_srvs"
                              "sensor_msgs"
                              "geometry_msgs"
                              "tf2"
                              "tf2_ros"
                              "tf2_eigen"
                              )

    ament_export_include_directories(${rclcpp_INCLUDE_DIRS}
                                     ${yak_INCLUDE_DIRS}
                                     )

    ament_export_dependencies(rclcpp yak tf2_eigen tf2)

    ament_package()
endif()

if ($ENV{ROS_VERSION} VERSION_EQUAL "1")
    set(ROS_LIB_DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
    set(ROS_BIN_DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
else()
    set(ROS_LIB_DESTINATION lib)
    set(ROS_BIN_DESTINATION bin)
endif()

install(TARGETS ${PROJECT_NAME}_node
        RUNTIME DESTINATION ${ROS_LIB_DESTINATION}
        )
