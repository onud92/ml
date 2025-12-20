# Check if ML_HELLO is set in Kconfig
if(CONFIG_ML_HELLO)

    # add source file directory
    zephyr_include_directories(.)

    # add wanted source files
    zephyr_library_sources(ml_hello.c)

endif()