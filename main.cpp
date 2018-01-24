#include "modelisation/OpenGL.h"
#include "analyse/EdgeDetection.h"
#include "modelisation/Transformation.h"
#include "physics/AngleModel.h"
#include "physics/CollisionDetection.h"

using namespace cv;
using namespace std;

CameraStream *cameraStream = nullptr;
OpenGL *window = nullptr;
AngleModel *angleModel = nullptr;
Ball *ball = nullptr;

/// Pour afficher les FPS
int frame = 0, myTime, timebase = 0;
double fps = 0.0;

/// Prototypes des fonctions de ce fichier
void loop(int);
void setupMaze();

int main(int argc, char** argv){

    bool anaglyph;
    ball = new Ball(0.5, 0.5, 0.02, 50);
    cameraStream = new CameraStream();
    namedWindow("aMAZEd Calibration");

    while(true){
        Mat currentFrame = cameraStream->getCurrentFrame();
        putText(currentFrame, "Press enter to play 3D mode", Point2i(0, currentFrame.rows - 50), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
        putText(currentFrame, "Press space bar to play normal mode", Point2i(0, currentFrame.rows), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
        float long12,long03, long01, long23, ratio = 0;
        EdgeDetection ED = EdgeDetection(currentFrame, true);
        vector<Point2i> coordCorner = ED.getCorner(currentFrame);
        long12 = sqrt(pow(coordCorner[1].x-coordCorner[2].x,2)+pow(coordCorner[1].y-coordCorner[2].y,2));
        long03 = sqrt(pow(coordCorner[0].x-coordCorner[3].x,2)+pow(coordCorner[0].y-coordCorner[3].y,2));

        if(long03 > long12*0.9 && long03 < long12*1.1){
            long01 = sqrt(pow(coordCorner[0].x-coordCorner[1].x,2)+pow(coordCorner[0].y-coordCorner[1].y,2));
            long23 = sqrt(pow(coordCorner[2].x-coordCorner[3].x,2)+pow(coordCorner[2].y-coordCorner[3].y,2));

            ratio = long01/long23;
            cout<<ratio<<endl;
        }
        bool is45 = ratio > 0.73 && ratio < 0.8 ;
        if(is45){
            putText(currentFrame, "O", Point2i(currentFrame.cols-20, 20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 0), 2);
        }else{
            putText(currentFrame, "X", Point2i(currentFrame.cols-20, 20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
        }
        imshow("aMAZEd Calibration", currentFrame);

        /// Si on appuie sur :
        /// touche espace => normal mode
        /// touche entrée => anaglyph mode
        int key = waitKey(30);
        if(key == 32 && is45){
            anaglyph = false;
            break;
        } else if(key == 13 && is45){
            anaglyph = true;
            break;
        }

    }

    Mat currentFrame = cameraStream->getCurrentFrame();
    double ratio = (double)currentFrame.cols / (double)currentFrame.rows;
    int width = 1000; /// Largeur de la fenêtre

    auto *glutMaster = new GlutMaster();
    window = new OpenGL(glutMaster, width, (int)(width / ratio), 0, 0, (char*)("aMAZEd"), ball, cameraStream, anaglyph);

    setupMaze();
    window->startTimer();

    destroyWindow("aMAZEd Calibration");
    glutMaster->CallGlutMainLoop();

    delete cameraStream;
    delete window;
    delete angleModel;

    return 0;
}

void loop(int endGame){

    if(endGame == 1){
        waitKey(0);
        exit(0);
    }

    /// Affichage FPS
    frame++;
    myTime = glutGet(GLUT_ELAPSED_TIME);
    if (myTime - timebase > 1000) {
        fps = frame * 1000.0 / (myTime - timebase);
        timebase = myTime;
        frame = 0;
    }
    window->setFps(fps);



    vector<Point2i> coordCorner;
    Mat currentFrame = cameraStream->getCurrentFrame();
    EdgeDetection edgeDetection = EdgeDetection(currentFrame, false);
    coordCorner = edgeDetection.getCorner(currentFrame);

    /// Si les 4 coins ont été détéctées
    if(coordCorner.size() == 4 && !edgeDetection.isReversed(coordCorner)) {
        Transformation transformation = Transformation(coordCorner, Size(currentFrame.cols, currentFrame.rows), 0.1, 20);
        angleModel->setCurrentTransformation(&transformation);

        vector<Wall> walls;
        if(CollisionDetection::findCollisions(ball, window->getWalls(), walls)){

            /// Detection de la nature de la collision
            bool verticalCollision = false;
            bool horizontalCollision = false;
            for(auto &wall : walls){
                if(!verticalCollision && wall.isVertical()) verticalCollision = true;
                if(!horizontalCollision && !wall.isVertical()) horizontalCollision = true;
            }

            /// Collision verticale on rebondit selon l'axe X
            if(verticalCollision){
                ball->setNextX(ball->getNextX() - ball->getVx());
                if(ball->getVx() > 0){
                    ball->setVx(-0.005);
                }else{
                    ball->setVx(0.005);
                }
                ball->setAx(0);
            }

            /// Collision horizontale on rebondit selon l'axe Y
            if(horizontalCollision){
                ball->setNextY(ball->getNextY() - ball->getVy());
                if(ball->getVy() > 0){
                    ball->setVy(-0.005);
                }else{
                    ball->setVy(0.005);
                }
                ball->setAy(0);
            }

            ball->updatePosition();

            /// S'il s'agit d'une collision sur le bout du mur
            if(CollisionDetection::findCollisions(ball, window->getWalls(), walls)){
                if(verticalCollision){
                    ball->setNextY(ball->getNextY() - ball->getVy() * 2);
                    if(ball->getVy() > 0){
                        ball->setVy(-0.005);
                    }else{
                        ball->setVy(0.005);
                    }
                    ball->setAy(0);
                }

                if(horizontalCollision){
                    ball->setNextX(ball->getNextX() - ball->getVx() * 2);
                    if(ball->getVx() > 0){
                        ball->setVx(-0.005);
                    }else{
                        ball->setVx(0.005);
                    }
                    ball->setAx(0);
                }
            }

        }else{
            ball->setAx(angleModel->getAngleY() / 10);
            ball->setAy(-angleModel->getAngleX() / 10);
            ball->updatePosition();
        }

        double p[16];
        double m[16];
        transformation.getProjectionMatrix(p);
        transformation.getModelviewMatrix(m);
        window->setProjectionMatrix(p);
        window->setModelviewMatrix(m);
    }

    glutPostRedisplay();

}

void setupMaze(){

    /// Calibration des couleurs


    Mat currentFrame = cameraStream->getCurrentFrame();
    EdgeDetection edgeDetection = EdgeDetection(currentFrame, true);

    vector<Point2i> coordCorner;
    vector<Point2i> coordStartEnd;
    vector<vector<Point2i>> lines;

    /// Tant que les 4 coins n'ont pas été détéctées
    do {

        currentFrame = cameraStream->getCurrentFrame();
        coordStartEnd = edgeDetection.startEndDetection(currentFrame);
        coordCorner = edgeDetection.getCorner(currentFrame);

        /// Detection des murs
        lines = edgeDetection.wallsDetection(currentFrame, coordCorner, coordStartEnd);

    }while(coordStartEnd.size() != 2);

    Transformation *transformation = new Transformation(coordCorner, Size(currentFrame.cols, currentFrame.rows), 1, 10);

    ///point d'arrivée sauvegarde
    Point2d *pointModelEnd = new Point2d(transformation->getModelPointFromImagePoint(coordStartEnd[1]));
    window->setEndPoint(pointModelEnd);

    ///set la boule aux coordonnées du départ détectés
    cv::Point2d pointModelStart = transformation->getModelPointFromImagePoint(coordStartEnd[0]);

    ///set la boule aux coordonnées du départ
    ball->setNextX(pointModelStart.x);
    ball->setNextY(pointModelStart.y);

    /// Calcul des coordonées des extrimités des murs
    vector<Wall> walls;
    for (const auto &line : lines) {

        Point2d pointImageA = transformation->getModelPointFromImagePoint(line[0]);
        Point2d pointImageB = transformation->getModelPointFromImagePoint(line[1]);

        Wall wall(pointImageA, pointImageB);

        walls.push_back(wall);
    }

    /// Murs extérieurs
//    walls.emplace_back(Point2d(0, 0), Point2d(0, 1));
//    walls.emplace_back(Point2d(1, 1), Point2d(0, 1));
//    walls.emplace_back(Point2d(1, 1), Point2d(1, 0));
//    walls.emplace_back(Point2d(1, 0), Point2d(0, 0));

    window->setWalls(walls);

    angleModel = new AngleModel(transformation);


}