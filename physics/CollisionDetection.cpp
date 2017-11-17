
#include <GL/glut.h>
#include <iostream>
#include "CollisionDetection.h"

using namespace std;

bool CollisionDetection::intersect(Ball *ball, Wall wall) {
    /// Calcul du vecteur leftNormal par rapport au mur
    /// --> v=[ax;ay] -> v_leftNorm = [-ay;ax]
    double x1 = wall.getStart().x;
    double y1 = wall.getStart().y;
    double x2 = wall.getEnd().x;
    double y2 = wall.getEnd().y;
    Vector2d leftNormal(-(y2-y1), (x2-x1));
    Vector2d line(x1,y1,x2,y2);

    /// Calcul de la distance perpendiculaire balle-mur
    Vector2d c1_circle (ball->getNextX() - x1, ball->getNextY() - y1);
    double perpendicular = c1_circle.projectionOn(leftNormal);

    line.sortPoints();
    /// Compare si la perpendiculaire est plus petite que le rayon de celui-ci + l'epaisseur des murs
    /// True si collision, sinon false
    if( fabs(perpendicular) <= ball->getR()+Wall::THICKNESS ) {
        return CollisionDetection::withinLine(ball, line) ;
    }

    return false;
}

bool CollisionDetection::withinLine(Ball *ball, Vector2d vect) {
    double xmin = vect.get_x1()-ball->getR();
    double xmax = vect.get_x2()+ball->getR();
    double ymin = vect.get_y1();
    double ymax = vect.get_y2();

    if(ymin > ymax) {
        double tmp = ymin;
        ymin = ymax;
        ymax = tmp;
    }

    ymin -= ball->getR();
    ymax += ball->getR();

    if(vect.get_x1() != vect.get_x2()) {

        /// si la position x est entre x1 et x2
        if(ball->getNextX() >= xmin && ball->getNextX() <= xmax) {
            /// Dans le cas ou y1 != y2
            if(vect.get_y1() != vect.get_y2()) {

                /// si la position y est entre y1 et y2
                if(ball->getNextY() >= ymin && ball->getNextY() <= ymax)
                    return true;
            }   /// Cas ou y1 = y2
            else {
                return true;
            }
        }
    }
        /// x1 = x2, forcement y1 != y2
    else {
        if(ball->getNextY() >= ymin && ball->getNextY() <= ymax)
            return true;
    }
    return false;
}

bool CollisionDetection::findCollisions(Ball *ball, vector<Wall> walls, vector<Wall> &touchedWalls) {
    bool result = false;
    for(Wall &current: walls) {
        if(intersect(ball, current)) {
            touchedWalls.push_back(current);
            result = true;
        }
    }
    return result;
}

bool CollisionDetection:: hasArrived(Ball *ball, cv::Point2d *point){
    return ball->getX() >= point->x - ball->getR()
           && ball->getX() <= point->x + ball->getR()
           && ball->getY() >= point->y - ball->getR()
           && ball->getY() <= point->y + ball->getR();
}


