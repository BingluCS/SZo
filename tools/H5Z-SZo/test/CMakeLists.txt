function(build_hdf5_test)
    get_filename_component(test_name ${ARGV0} NAME_WE)
    add_executable(${test_name} ${ARGV})
    target_link_libraries(${test_name} PUBLIC SZo hdf5szo)
endfunction(build_hdf5_test)

build_hdf5_test(szoToHDF5.cpp)
build_hdf5_test(HDF5ToSzo.cpp)
build_hdf5_test(convertBinToHDF5.cpp)
build_hdf5_test(cdvalueHelper.cpp)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/h5repack.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})