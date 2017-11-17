#ifndef AMAZED_FLUXCAMERA_H
#define AMAZED_FLUXCAMERA_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>


class CameraStream {
private:
    cv::VideoCapture capture;
public:
    CameraStream();
    bool isOpen();
    void showCamera();
    cv::Mat getCurrentFrame();
};


#endif //AMAZED_FLUXCAMERA_H
