#
# Tyler Filla
# CS 4340 - Project 5
# December 13, 2018
#

cmake_minimum_required(VERSION 3.8)
project(CS4340P5)

set(p5_SRC_FILES
        src/main.cpp
        )

add_executable(p5 ${p5_SRC_FILES})
set_target_properties(p5 PROPERTIES CXX_STANDARD 17)
