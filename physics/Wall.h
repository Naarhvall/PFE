//
// Created by Alexandre on 23/10/2017.
//

#ifndef AMAZED_WALL_H
#define AMAZED_WALL_H


#include <opencv2/core/types.hpp>

class Wall {

private:

    /// Hauteur des murs
    double const WALL_HEIGHT = -0.05f;

    bool vertical;

    cv::Point2d start;
    cv::Point2d end;

public:

    double constexpr static THICKNESS = 0.02;

    Wall();
    Wall(const Wall &wall);
    Wall(cv::Point2d start, cv::Point2d end);

    Wall& operator=(const Wall& wall);

    void setStart(const cv::Point2d &start);
    void setEnd(const cv::Point2d &end);
    const cv::Point2d &getStart() const;
    const cv::Point2d &getEnd() const;
    bool isVertical() const;

    void draw();

};


#endif //AMAZED_WALL_H
