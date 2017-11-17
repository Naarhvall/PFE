#include "Ball.h"

Ball::Ball(double x, double y, double r, GLint m) {
    this->x = x;
    this->y = y;
    this->z = -r;
    this->nextX = x;
    this->nextY = y;
    this->nextZ = -r;
    this->r = r;
    this->m = m;
    this->vx = 0;
    this->vy = 0;
    this->vz = 0;
    this->ax = 0;
    this->ay = 0;
    this->az = 0;
}

double Ball::getX() const {
    return x;
}

void Ball::setX(double x) {
    Ball::x = x;
}

double Ball::getY() const {
    return y;
}

void Ball::setY(double y) {
    Ball::y = y;
}

double Ball::getZ() const {
    return z;
}

void Ball::setZ(double z) {
    Ball::z = z;
}

double Ball::getR() const {
    return r;
}

double Ball::getAx() const {
    return ax;
}
void Ball::setAx(double a) {
    Ball::ax = a;
}

double Ball::getAy() const {
    return ay;
}
void Ball::setAy(double a) {
    Ball::ay = a;
}

double Ball::getAz() const {
    return az;
}
void Ball::setAz(double a) {
    Ball::az = a;
}

double Ball::getVx() const {
    return vx;
}

void Ball::setVx(double vx) {
    Ball::vx = vx;
}

double Ball::getVy() const {
    return vy;
}

void Ball::setVy(double vy) {
    Ball::vy = vy;
}

double Ball::getVz() const {
    return vz;
}

void Ball::setVz(double vz) {
    Ball::vz = vz;
}

GLint Ball::getM() const {
    return m;
}

void Ball::draw(){
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslated(this->x, this->y, this->z);
    glutSolidSphere(this->r, this->m, this->m);
    glPopMatrix();
}

void Ball::updatePosition() {

    vx += ax;
    vy += ay;

    if(vx > 0.03) vx = 0.03;
    if(vx < -0.03) vx = -0.03;
    if(vy > 0.03) vy = 0.03;
    if(vy < -0.03) vy = -0.03;

    if(nextX > 1.0f) nextX = 1.0f;
    if(nextX < 0.0f) nextX = 0.0f;
    if(nextY > 1.0f) nextY = 1.0f;
    if(nextY < 0.0f) nextY = 0.0f;


    x = nextX;
    y = nextY;

    nextX += vx;
    nextY += vy;


}

double Ball::getNextX() const {
    return nextX;
}

void Ball::setNextX(double nextX) {
    Ball::nextX = nextX;
}

double Ball::getNextY() const {
    return nextY;
}

void Ball::setNextY(double nextY) {
    Ball::nextY = nextY;
}

double Ball::getNextZ() const {
    return nextZ;
}

void Ball::setNextZ(double nextZ) {
    Ball::nextZ = nextZ;
}
