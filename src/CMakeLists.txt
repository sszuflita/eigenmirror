cmake_minimum_required(VERSION 2.8)
project( Eigenfaces )
find_package( OpenCV REQUIRED )
add_executable( DisplayImage DisplayImage.cpp)
add_executable( camera camera.cpp)
target_link_libraries( DisplayImage ${OpenCV_LIBS})
target_link_libraries( camera ${OpenCV_LIBS})

