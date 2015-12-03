cmake_minimum_required(VERSION 2.8)

project(ml2048)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -g")

add_executable(game game.cpp)
add_library(grid utils.cpp grid.cpp)

add_library(neuron ../CNeuralNets/FeedforwardNet/neuron.cpp ../CNeuralNets/FeedforwardNet/neuron.h)
add_library(layer ../CNeuralNets/FeedforwardNet/layer.cpp ../CNeuralNets/FeedforwardNet/layer.h)
add_library(network ../CNeuralNets/FeedforwardNet/network.cpp ../CNeuralNets/FeedforwardNet/network.h)
add_library(brain ../CNeuralNets/FeedforwardNet/brain.cpp ../CNeuralNets/FeedforwardNet/brain.h)

add_executable(ml qlearn.cpp)

target_link_libraries(game grid)
target_link_libraries(ml brain network layer neuron grid)