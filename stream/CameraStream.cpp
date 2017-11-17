#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "CameraStream.h"

using namespace std;
using namespace cv;

CameraStream::CameraStream(){
    this->capture = VideoCapture(0);
}

bool CameraStream::isOpen(){
   return this->capture.isOpened();
}

void CameraStream::showCamera() {
    namedWindow("Capture",WINDOW_AUTOSIZE);
    Mat currentFrame;
    while(true){
        this->capture >> currentFrame; // get a new frame from camera
        imshow("Capture", currentFrame);

        // Quitte l'application lorsqu'on appuie sur Echap
        if(waitKey(30) == 27) break;
    }
}

Mat CameraStream::getCurrentFrame() {
    Mat currentFrame;
    this->capture >> currentFrame; // get a new frame from camera
    Mat frameFlip;
    flip(currentFrame, frameFlip, 1); // 1 -> flip axe y (0 -> axe x)
    return frameFlip;
}