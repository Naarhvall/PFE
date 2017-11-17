#include <GL/gl.h>
#include "Wall.h"

using namespace cv;

Wall::Wall() = default;

void Wall::draw() {

    if(vertical){

        glBegin(GL_POLYGON);
        glTexCoord2d(0, 0); glVertex3d(start.x - THICKNESS / 2, start.y, 0);            // A
        glTexCoord2d(1, 0); glVertex3d(end.x - THICKNESS / 2, end.y, 0);                // B
        glTexCoord2d(1, 1); glVertex3d(end.x - THICKNESS / 2, end.y, WALL_HEIGHT);    // C
        glTexCoord2d(0, 1); glVertex3d(start.x - THICKNESS / 2, start.y, WALL_HEIGHT);// D
        glEnd();

        glBegin(GL_POLYGON);
        glTexCoord2d(0, 0); glVertex3d(start.x + THICKNESS / 2, start.y, 0);            // E
        glTexCoord2d(1, 0); glVertex3d(end.x + THICKNESS / 2, end.y, 0);                // F
        glTexCoord2d(1, 1); glVertex3d(end.x + THICKNESS / 2, end.y, WALL_HEIGHT);    // G
        glTexCoord2d(0, 1); glVertex3d(start.x + THICKNESS / 2, start.y, WALL_HEIGHT);// H
        glEnd();

        glBegin(GL_POLYGON);
        glTexCoord2d(1, 0); glVertex3d(end.x - THICKNESS / 2, end.y, 0);                // B
        glTexCoord2d(1, 1); glVertex3d(end.x - THICKNESS / 2, end.y, WALL_HEIGHT);    // C
        glTexCoord2d(1, 1); glVertex3d(end.x + THICKNESS / 2, end.y, WALL_HEIGHT);    // G
        glTexCoord2d(1, 0); glVertex3d(end.x + THICKNESS / 2, end.y, 0);                // F
        glEnd();

        glBegin(GL_POLYGON);
        glTexCoord2d(0, 0); glVertex3d(start.x - THICKNESS / 2, start.y, 0);            // A
        glTexCoord2d(0, 1); glVertex3d(start.x - THICKNESS / 2, start.y, WALL_HEIGHT);// D
        glTexCoord2d(0, 1); glVertex3d(start.x + THICKNESS / 2, start.y, WALL_HEIGHT);// H
        glTexCoord2d(0, 0); glVertex3d(start.x + THICKNESS / 2, start.y, 0);            // E
        glEnd();

        glBegin(GL_POLYGON);
        glTexCoord2d(0.8, 1); glVertex3d(start.x - THICKNESS / 2, start.y, WALL_HEIGHT);// D
        glTexCoord2d(1, 1); glVertex3d(end.x - THICKNESS / 2, end.y, WALL_HEIGHT);    // C
        glTexCoord2d(1, 1); glVertex3d(end.x + THICKNESS / 2, end.y, WALL_HEIGHT);    // G
        glTexCoord2d(0.8, 1); glVertex3d(start.x + THICKNESS / 2, start.y, WALL_HEIGHT);// H
        glEnd();

    }else{

        glBegin(GL_POLYGON);
        glTexCoord2d(0, 0); glVertex3d(start.x , start.y - THICKNESS / 2, 0);           // A
        glTexCoord2d(1, 0); glVertex3d(end.x, end.y - THICKNESS / 2, 0);                // B
        glTexCoord2d(1, 1); glVertex3d(end.x, end.y - THICKNESS / 2, WALL_HEIGHT);    // C
        glTexCoord2d(0, 1); glVertex3d(start.x, start.y - THICKNESS / 2, WALL_HEIGHT);// D
        glEnd();

        glBegin(GL_POLYGON);
        glTexCoord2d(0, 0); glVertex3d(start.x, start.y + THICKNESS / 2, 0);            // E
        glTexCoord2d(1, 0); glVertex3d(end.x, end.y + THICKNESS / 2, 0);                // F
        glTexCoord2d(1, 1); glVertex3d(end.x, end.y + THICKNESS / 2, WALL_HEIGHT);    // G
        glTexCoord2d(0, 1); glVertex3d(start.x, start.y + THICKNESS / 2, WALL_HEIGHT);// H
        glEnd();

        glBegin(GL_POLYGON);
        glTexCoord2d(1, 0); glVertex3d(end.x, end.y - THICKNESS / 2, 0);                // B
        glTexCoord2d(1, 1); glVertex3d(end.x, end.y - THICKNESS / 2, WALL_HEIGHT);    // C
        glTexCoord2d(1, 1); glVertex3d(end.x, end.y + THICKNESS / 2, WALL_HEIGHT);    // G
        glTexCoord2d(1, 0); glVertex3d(end.x, end.y + THICKNESS / 2, 0);                // F
        glEnd();

        glBegin(GL_POLYGON);
        glTexCoord2d(0, 0); glVertex3d(start.x, start.y - THICKNESS / 2, 0);            // A
        glTexCoord2d(0, 1); glVertex3d(start.x, start.y - THICKNESS / 2, WALL_HEIGHT);// D
        glTexCoord2d(0, 1); glVertex3d(start.x, start.y + THICKNESS / 2, WALL_HEIGHT);// H
        glTexCoord2d(0, 0); glVertex3d(start.x, start.y + THICKNESS / 2, 0);            // E
        glEnd();

        glBegin(GL_POLYGON);
        glTexCoord2d(0.8, 1); glVertex3d(start.x, start.y - THICKNESS / 2, WALL_HEIGHT);// D
        glTexCoord2d(1, 0.2); glVertex3d(end.x, end.y - THICKNESS / 2, WALL_HEIGHT);    // C
        glTexCoord2d(1, 0.2); glVertex3d(end.x, end.y + THICKNESS / 2, WALL_HEIGHT);    // G
        glTexCoord2d(0.8, 1); glVertex3d(start.x, start.y + THICKNESS / 2, WALL_HEIGHT);// H
        glEnd();

    }

}

Wall::Wall(cv::Point2d start, cv::Point2d end) {
    this->start = start;
    this->end = end;
    this->vertical = (fabs(start.x - end.x) < fabs(start.y - end.y));
}

const Point2d &Wall::getStart() const {
    return start;
}

const Point2d &Wall::getEnd() const {
    return end;
}

void Wall::setStart(const Point2d &start) {
    Wall::start = start;
}

void Wall::setEnd(const Point2d &end) {
    Wall::end = end;
}

bool Wall::isVertical() const {
    return vertical;
}

Wall& Wall::operator=(const Wall& wall) {
    this->start = wall.getStart();
    this->end = wall.getEnd();
}

Wall::Wall(const Wall &wall) {
    this->start = wall.getStart();
    this->end = wall.getEnd();
    this->vertical = wall.isVertical();
}
