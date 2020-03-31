# ADAS-Features

Advance Driver Assistance System

FVSA - Forward Vehicle Start Alarm
Here i am using OpenCv-4 C++ library for Frontal Vechicle Start Alarm scenario. 

First compile opencv library with contrib modules and for running fvsa on ubuntu system check the compile options:

Compile options :  g++ -std=c++11  fvsa.cpp -o fvsa -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_features2d  -lopencv_core -lopencv_objdetect -lopencv_tracking

  
TLDS - Traffic Light Detection System
TLDS function is made to detect changes in traffic light i.e.When color changes from Red to Green.
TLDS function can vary from country to country because of different traffic light patterns. 
In here we provide this fucntion for South Korea and Australia.
