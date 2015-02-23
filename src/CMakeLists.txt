cmake_minimum_required(VERSION 2.8)

project( Eigenfaces )

find_package( OpenCV REQUIRED )
find_package(Boost COMPONENTS regex system filesystem REQUIRED)

include_directories(${Boost_INCLUDE_DIRS}) 
add_executable( ComputeEigenfaces ComputeEigenfaces.cpp)
target_link_libraries(ComputeEigenfaces
	${OpenCV_LIBS}
	${Boost_FILESYSTEM_LIBRARY}
	${Boost_SYSTEM_LIBRARY}
	${Boost_REGEX_LIBRARY})


add_executable( Stream Stream.cpp)
target_link_libraries(Stream
	${OpenCV_LIBS}
	${Boost_FILESYSTEM_LIBRARY}
	${Boost_SYSTEM_LIBRARY}
	${Boost_REGEX_LIBRARY})

