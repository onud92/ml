cmake_minimum_required(VERSION 3.20.0)

# Zephyr Umgebung laden
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(my_ml)

# Quellcode hinzufügen
target_sources(app PRIVATE src/main.c)