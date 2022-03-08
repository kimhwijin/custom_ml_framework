cmake_minimum_required(VERSION 3.21)
project(c__)

set(CMAKE_CXX_STANDARD 17)

add_executable(c__ abalone/AbaloneExecute.cpp abalone/LoadCsvDataset.cpp abalone/LoadCsvDataset.h abalone/InitModel.cpp main.cpp)