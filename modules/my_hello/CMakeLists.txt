# Check if ML_HELLO is set in Kconfig
if(CONFIG_MY_HELLO)

    # add source file directory
    zephyr_include_directories(.)

    # add wanted source files
    zephyr_library_sources(my_hello.c)

endif()