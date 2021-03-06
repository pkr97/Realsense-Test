# Realsense-Test


# @@ Installation of Intel realsense

# Add Intel server to the list of repositories :
1. echo 'deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main' | sudo tee /etc/apt/sources.list.d/realsense-public.list

# It is recommended to backup /etc/apt/sources.list.d/realsense-public.list file in case of an upgrade.

# Register the server’s public key :
2. sudo apt-key adv --keyserver keys.gnupg.net --recv-key 6F3EFCDE

# Refresh the list of repositories and packages available :
3. sudo apt-get update

# In order to run demos install:
4. sudo apt-get install librealsense2-dkms
5. sudo apt-get install librealsense2-utils
# The above two lines will deploy librealsense2 udev rules, kernel drivers, runtime library and executable demos and tools. 

# Reconnect the Intel RealSense depth camera and run: realsense-viewer

# Developers shall install additional packages:
6. sudo apt-get install librealsense2-dev
7. sudo apt-get install librealsense2-dbg
# With dev package installed, you can compile an application with librealsense using g++ -std=c++11 filename.cpp -lrealsense2 or an IDE of your choice.

# Verify that the kernel is updated :
8. modinfo uvcvideo | grep "version:"
# should include 'realsense' string

# before example compilation
9. sudo apt-get install libglfw3-dev

# download the older librealsense directory.
10. git clone https://github.com/IntelRealSense/librealsense/archive/v2.10.3.zip

# Unzip the downloaded file
11. unzip librealsense-2.10.3.zip

## @@ Install Opencv 3.4 using the instructions given in OpenCV_3.4_Installation.txt 

# compile intel realsense source code
12. cd librealsense-2.10.3
 @ @ modify the main CMakelists.txt.
 
 find the line "option(BUILD_CV_EXAMPLES "Build OpenCV examples" OFF)"
 change to "option(BUILD_CV_EXAMPLES "Build OpenCV examples" ON)"

# Build librealsense source code
13. mkdir build
14. cd build

# Linking
15. cmake ..

# Compiling
16. make -j4


