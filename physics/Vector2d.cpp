
#include <cmath>
#include "Vector2d.h"

Vector2d::Vector2d(double x1, double y1, double x2, double y2) {
    this->_x1 = x1;
    this->_y1 = y1;
    this->_x2 = x2;
    this->_y2 = y2;
    this->_X = x2-x1;
    this->_Y = y2-y1;
}

Vector2d::Vector2d(double X, double Y) {
    this->_X = X;
    this->_Y = Y;
}

void Vector2d::sortPoints(){
    if(_x1 > _x2 ) {
        double tmp = _x1;
        _x1 = _x2;
        _x2 = tmp;

        tmp = _y1;
        _y1 = _y2;
        _y2 = tmp;
    }
    else if (_x1 == _x2) {
        if(_y1 > _y2) {                 //x1 = x2 donc permutation inutile pour les x
            double tmp = _y1;
            _y1 = _y2;
            _y2 = tmp;
        }
    }
}

double Vector2d::projectionOn(Vector2d v2) {
    return this->dotProduct(v2.normalize());
}

void Vector2d::setMagnitude(double magnitude) {
    double angle_current = atan2(this->_Y, this->_X);
    this->polar(magnitude, angle_current);
}

double Vector2d::getMagnitude() {
    return sqrt( this->get_X()*this->get_X() + this->get_Y()*this->get_Y() );
}

/// A priori une division par 0 ne devrait pas arriver
/// cela voudrait dire qu'un mur a X = Y, donc un point
Vector2d Vector2d::normalize() {
    double magnitude = this->getMagnitude();
    Vector2d normalizedVect(this->get_X()/magnitude, this->get_Y()/magnitude);
    return normalizedVect;
}

double Vector2d::dotProduct(Vector2d vect) {
    //             x1     *    x2       +     y1       *    y2
    return this->get_X()*vect.get_X() + this->get_Y()*vect.get_Y();
}

void Vector2d::polar(double magnitude, double angleRadian) {
    this->_X = magnitude * cos(angleRadian);
    this->_Y = magnitude * sin(angleRadian);

}

double Vector2d::get_x1() const {
    return _x1;
}

void Vector2d::set_x1(double _x1) {
    Vector2d::_x1 = _x1;
}

double Vector2d::get_y1() const {
    return _y1;
}

void Vector2d::set_y1(double _y1) {
    Vector2d::_y1 = _y1;
}

double Vector2d::get_x2() const {
    return _x2;
}

void Vector2d::set_x2(double _x2) {
    Vector2d::_x2 = _x2;
}

double Vector2d::get_y2() const {
    return _y2;
}

void Vector2d::set_y2(double _y2) {
    Vector2d::_y2 = _y2;
}

double Vector2d::get_X() const {
    return _X;
}

void Vector2d::set_X(double _X) {
    Vector2d::_X = _X;
}

double Vector2d::get_Y() const {
    return _Y;
}

void Vector2d::set_Y(double _Y) {
    Vector2d::_Y = _Y;
}


