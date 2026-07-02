set(SZO_SOURCE_FILE "${CMAKE_SOURCE_DIR}/tools/sz3/szo.cpp")
set(SZO_TARGET_NAME szo)

# Vendored standalone FSE (entropy coder) — compiled into szo directly so the
# FSE_* symbols come from our own source, NOT from libzstd. Hidden visibility
# keeps them out of the dynamic symbol table so they neither clash with nor
# interpose libzstd's own FSE_* (libzstd is still linked for the ZSTD_* wrap).
file(GLOB ZFSE_SOURCES "${CMAKE_SOURCE_DIR}/include/SZo/encoder/zfse/*.c")
set_source_files_properties(${ZFSE_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-fvisibility=hidden")

message(STATUS "Target = ${SZO_TARGET_NAME}")
add_executable(${SZO_TARGET_NAME} ${SZO_SOURCE_FILE} ${ZFSE_SOURCES})
target_link_libraries(${SZO_TARGET_NAME} PRIVATE SZo)
install(TARGETS ${SZO_TARGET_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES testfloat_8_8_128.dat DESTINATION ${CMAKE_INSTALL_DATADIR}/SZo)

#add_subdirectory(demo)
