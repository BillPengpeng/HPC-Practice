
opencv_include=../include/linux/
opencv_lib=../third_party/linux/

# speed_rgb2gray
export LD_LIBRARY_PATH=$opencv_lib:$LD_LIBRARY_PATH
obj=speed_rgb2gray_linux
g++ -O2 -march=native ../src/speed_rgb2gray.cpp -I $opencv_include -I $opencv_include/opencv2 -DLINUX \
    -L $opencv_lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lpthread -o ../build/$obj
../build/$obj