#Either use:
#find_package(HDF5 1.12.1 REQUIRED)
#or manually set the hdf5 paths:
#set(HDF5_LIBRARIES "/path/to/hdf5-1.12.1/hdf5/lib/libhdf5.so")
#set(HDF5_INCLUDE_DIRS "/path/to/hdf5-1.12.1/hdf5/include")

find_package(HDF5 COMPONENTS C REQUIRED)

#message("HDF5LIB PATH:" ${HDF5_LIBRARIES})
#message("HDF5INCLUDE PATH:" ${HDF5_INCLUDE_DIRS})

# Vendored standalone FSE (entropy coder) — compiled into the filter so FSE_*
# resolve to our own source: system libzstd caps FSE_MAX_TABLELOG at 11, while
# the o1tans context tables use tableLog 13. Hidden visibility keeps the FSE_*
# symbols local to this .so (internal calls bind at link time, no libzstd
# interposition, nothing exported).
file(GLOB ZFSE_SOURCES "${CMAKE_SOURCE_DIR}/include/SZo/encoder/zfse/*.c")
set_source_files_properties(${ZFSE_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-fvisibility=hidden")

add_library(hdf5szo
        src/H5Z_SZo.cpp
        ${ZFSE_SOURCES}
)

target_link_libraries(hdf5szo
        PUBLIC SZo HDF5::HDF5
)

# The compression happens inside this filter TU, so it must be built with the SIMD
# flags for the AVX2/SVE2 code paths in the (header-only) SZo library to activate.
# Linking SZo (INTERFACE) already propagates these, but set them explicitly to match
# the standalone szo binary and the SZOc C API.
if (ENABLE_AVX2)
    target_compile_options(hdf5szo PRIVATE -mavx2 -mfma)
endif ()
if (ENABLE_SVE2)
    target_compile_options(hdf5szo PRIVATE -march=armv8.6-a+sve2)
endif ()

if (${HDF5_IS_PARALLEL})
    find_package(MPI REQUIRED)
    target_link_libraries(hdf5szo
            PUBLIC MPI::MPI_CXX)
endif ()

target_include_directories(hdf5szo
        PUBLIC
        ${HDF5_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:hdf5_szo>
)

add_subdirectory(test)

install(TARGETS hdf5szo EXPORT HDF5SZo
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hdf5_szo)
export(TARGETS hdf5szo FILE HDF5SZo.cmake)
