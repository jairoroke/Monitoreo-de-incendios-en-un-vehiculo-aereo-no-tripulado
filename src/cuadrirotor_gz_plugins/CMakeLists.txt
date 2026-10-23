cmake_minimum_required(VERSION 3.16)
project(cuadrirotor_gz_plugins)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin REQUIRED COMPONENTS register)
find_package(gz_transport_vendor REQUIRED)
find_package(gz-transport REQUIRED)
find_package(gz_msgs_vendor REQUIRED)
find_package(gz-msgs REQUIRED)

add_library(FuerzaViento SHARED src/FuerzaViento.cc)
target_link_libraries(FuerzaViento
  gz-sim::gz-sim gz-plugin::register gz-transport::gz-transport gz-msgs::gz-msgs)

add_library(ComandosMotores SHARED src/ComandosMotores.cc)
target_link_libraries(ComandosMotores
  gz-sim::gz-sim gz-plugin::register gz-transport::gz-transport gz-msgs::gz-msgs)

install(TARGETS FuerzaViento ComandosMotores LIBRARY DESTINATION lib)

# Agrega lib/ a GZ_SIM_SYSTEM_PLUGIN_PATH al hacer `source install/setup.bash`.
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")

ament_package()
