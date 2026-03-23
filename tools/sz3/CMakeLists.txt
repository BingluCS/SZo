set(SZO_SOURCE_FILE "${CMAKE_SOURCE_DIR}/tools/sz3/szo.cpp")
set(SZO_TARGET_NAME szo)

message(STATUS "Target = ${SZO_TARGET_NAME}")
add_executable(${SZO_TARGET_NAME} ${SZO_SOURCE_FILE})
target_link_libraries(${SZO_TARGET_NAME} PRIVATE SZ3)
install(TARGETS ${SZO_TARGET_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES testfloat_8_8_128.dat DESTINATION ${CMAKE_INSTALL_DATADIR}/SZ3)

#add_subdirectory(demo)
