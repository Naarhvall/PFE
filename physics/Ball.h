#ifndef AMAZED_BALL_H
#define AMAZED_BALL_H

#include <GL/freeglut.h>

class Ball  {
private:
    double nextX;
    double nextY;
    double nextZ;
    double x;

private:
    //position selon l'axe x
    double y;    //position selon l'axe y
    double z;    //position selon l'axe z
    double r;    //rayon de la boule
    double ax;   //acceleration selon l'axe x
    double ay;   //acceleration selon l'axe y
    double az;   //acceleration selon l'axe z
    double vx;   //vitesse selon l'axe x
    double vy;   //vitesse selon l'axe y
    double vz;   //vitesse selon l'axe z
    GLint m;    //maillage

public:
    Ball(double x, double y, double r, GLint m);

    double getX() const;

    void setX(double x);

    double getY() const;

    void setY(double y);

    double getZ() const;

    void setZ(double z);

    double getR() const;

    double getAx() const;

    void setAx(double a);

    double getAy() const;

    void setAy(double a);

    double getAz() const;

    void setAz(double a);

    double getVx() const;

    void setVx(double vx);

    double getVy() const;

    void setVy(double vy);

    double getVz() const;

    void setVz(double vz);

    GLint getM() const;

    double getNextX() const;

    void setNextX(double nextX);

    double getNextY() const;

    void setNextY(double nextY);

    double getNextZ() const;

    void setNextZ(double nextZ);

    void draw();

    void updatePosition();
};

#endif //AMAZED_BALL_H
