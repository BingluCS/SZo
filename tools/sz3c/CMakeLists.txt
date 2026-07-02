set(SZOC_TARGET_NAME SZOc)

# Match the standalone szo binary: RleFseEncoder needs the vendored FSE
# implementation linked into this shared library as well.
file(GLOB SZOC_ZFSE_SOURCES "${CMAKE_SOURCE_DIR}/include/SZo/encoder/zfse/*.c")
set_source_files_properties(${SZOC_ZFSE_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-fvisibility=hidden")

add_library(${SZOC_TARGET_NAME} src/szo.cpp ${SZOC_ZFSE_SOURCES})

target_link_libraries(${SZOC_TARGET_NAME} PUBLIC SZo)

# Keep the C API shared library on the same SIMD code path as the top-level build.
if(ENABLE_AVX2)
    target_compile_options(${SZOC_TARGET_NAME} PRIVATE -mavx2 -mfma)
endif()

if(ENABLE_SVE2)
    target_compile_options(${SZOC_TARGET_NAME} PRIVATE -march=armv8.6-a+sve2)
endif()

target_compile_definitions(${SZOC_TARGET_NAME} PRIVATE SZOC_EXPORTS)

target_include_directories(
        ${SZOC_TARGET_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/SZOc>
)

install(TARGETS ${SZOC_TARGET_NAME}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/SZOc)
