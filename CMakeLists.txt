cmake_minimum_required(VERSION 2.8)

project(ml2048)

set(netDir ./CPPReinforcementLearning/FeedforwardNet/)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -g")

add_executable(game game.cpp)
add_library(grid utils.cpp grid.cpp)

add_library(neuron ${netDir}/neuron.cpp ${netDir}/neuron.h)
add_library(layer ${netDir}/layer.cpp ${netDir}/layer.h)
add_library(network ${netDir}/network.cpp ${netDir}/network.h)
add_library(brain ${netDir}/brain.cpp ${netDir}/brain.h)

add_executable(ml qlearn.cpp)

target_link_libraries(game grid)
target_link_libraries(ml brain network layer neuron grid)