
opencv_include=/usr/local/include/opencv4
opencv_lib=/usr/local/lib

# speed_rgb2gray
export LD_LIBRARY_PATH=$opencv_lib:$LD_LIBRARY_PATH
#echo $LD_LIBRARY_PATH
g++ -O2 -march=native ../src/speed_rgb2gray.cpp -I $opencv_include -I $opencv_include/opencv2 \
    -L $opencv_lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lpthread -o ../build/speed_rgb2gray 
../build/speed_rgb2gray