if(SYCL_CTS_ENABLE_EXT_USE_PINNED_HOST_MEMORY_TESTS)
    file(GLOB test_cases_list *.cpp)

    add_cts_test(${test_cases_list})
endif()
