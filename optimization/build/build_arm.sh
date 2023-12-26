
opencv_include=/home/chenpeng/opencv-4.5.0/arm_build/install/include/opencv4
opencv_lib=/home/chenpeng/opencv-4.5.0/arm_build/install/lib

# speed_rgb2gray
export LD_LIBRARY_PATH=$opencv_lib:$LD_LIBRARY_PATH
obj=speed_rgb2gray_armv7
arm-linux-gnueabihf-g++ -O2  -march=armv7 ../src/speed_rgb2gray.cpp -I $opencv_include -I $opencv_include/opencv2 -L$opencv_lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -o ../build/$obj  
#../build/speed_rgb2gray