find_package(GTest REQUIRED)

aux_source_directory(./ DIR_TEST)

add_executable(test_llm ${DIR_TEST})
target_include_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/kuiper/include)
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_llm llama gtest gtest_main)