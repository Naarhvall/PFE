#include "AngleModel.h"

using namespace cv;

AngleModel::AngleModel(Transformation *initialTransformation) {
    this->initialTransformation = initialTransformation;
}

void AngleModel::setCurrentTransformation(Transformation *currentTransformation) {
    this->currentTransformation = currentTransformation;
    computeAngle();
}

double AngleModel::scalarProduct(Mat const &a, Mat const &b) const{
    return  a.at<double>(0) * b.at<double>(0) +
            a.at<double>(1) * b.at<double>(1) +
            a.at<double>(2) * b.at<double>(2);
}

double AngleModel::crossProduct(cv::Mat const &a, cv::Mat const &b) const {
    return a.at<double>(1) * b.at<double>(2) - a.at<double>(2) * b.at<double>(1) +
           b.at<double>(0) * a.at<double>(2) - a.at<double>(0) * b.at<double>(2) +
           a.at<double>(0) * b.at<double>(1) - a.at<double>(1) * b.at<double>(0);
}

double AngleModel::norm(cv::Mat const &a) const{
    return sqrt(
            a.at<double>(0) * a.at<double>(0) +
            a.at<double>(1) * a.at<double>(1) +
            a.at<double>(2) * a.at<double>(2)
    );
}

double AngleModel::angleOriente(cv::Mat &a, cv::Mat &b) const {

    double normal = norm(a) * norm(b);

    double sinus = crossProduct(a, b) / normal;
    double cosinus = scalarProduct(a, b) / normal;

    if(sinus > 0){
        return acos(cosinus);
    }else{
        return -acos(cosinus);
    }

}

void AngleModel::computeAngle() {

    Vec3d initialEuler = initialTransformation->getEulerAngle();
    Vec3d currentEuler = currentTransformation->getEulerAngle();

    this->angleX = initialEuler[0] - currentEuler[0];
    this->angleY = initialEuler[1] - currentEuler[1];
    this->angleZ = initialEuler[2] - currentEuler[2];


}

double AngleModel::getAngleX() const {
    return angleX;
}

double AngleModel::getAngleY() const {
    return angleY;
}

double AngleModel::getAngleZ() const {
    return angleZ;
}