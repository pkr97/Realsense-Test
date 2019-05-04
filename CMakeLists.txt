cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )


add_executable( DisplayImage DisplayImage.cpp ) 
add_executable( ReadVideo ReadVideo.cpp )

target_link_libraries( DisplayImage ${OpenCV_LIBS} )
target_link_libraries( ReadVideo ${OpenCV_LIBS} )
