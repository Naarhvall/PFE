
#ifndef AMAZED_COLLISIONDETECTION_H
#define AMAZED_COLLISIONDETECTION_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Ball.h"
#include "Vector2d.h"
#include "Wall.h"

class CollisionDetection {
private :

public  :
    /**
     * Le vecteur doit etre pre-traite avec sortPoints
     * Retourne vrai si la position x et y du cercle
     * est comprise entre le x1;y1 et x2;y2 du vecteur
     */
    static bool withinLine(Ball *ball, Vector2d vect);

    static bool hasArrived(Ball *ball, cv::Point2d *point);

    /**
     * Detecte si il y a une collision entre une ligne (+ une epaisseur) et la balle
     * @param ball
     * @param wall
     * @return True si collision, false sinon
     */
    static bool intersect(Ball *ball, Wall wall);

    /**
     *  Parcours tous les murs (walls) et range dans touchedWalls les murs qui touchent la balle.
     *  La fonction renvoie false si aucun mur a ete touche, true sinon
     */
    static bool findCollisions(Ball *ball, std::vector<Wall> walls, std::vector<Wall> &touchedWalls);

};


#endif //AMAZED_COLLISIONDETECTION_H
