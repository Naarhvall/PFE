#include <iostream>
#include "EdgeDetection.h"
#include "opencv2/opencv.hpp"
#include "../stream/CameraStream.h"

using namespace std;
using namespace cv;

///Coordonnées du point permettant la calibration pour la mask sur le niveau de gris
int StartingPointX = 0 ;
int StartingPointY = 0;


///Fonction permettant la calibration de la couleur
Mat EdgeDetection::colorCalibration(Mat img){
    ///Initialisation des varaibles
    std::vector<Point2i> keypoints;
    Mat imgGrey;
    cvtColor(img, imgGrey, COLOR_RGB2GRAY);

    ///On initialise le point de calibration au milieu de l'écran
    if(StartingPointX == 0 && StartingPointY == 0){
        StartingPointX = imgGrey.cols / 2 ;
        StartingPointY = imgGrey.rows / 2 ;
    }


    ///Création d'un mask permetant de sélectionner uniquement les 4 coins
    Mat mask;

    ///On récupère le niveau de gris du pixel du mileu (à changer)
    auto midGrey = (int)imgGrey.at<uchar>(StartingPointY, StartingPointX);
    ///On créé le mask en fonction du niveau de gris précédent
    inRange(imgGrey, midGrey-40, midGrey+40, mask);

    ///On manipule le mask afin de ne récupérer que les 4 coins
    Mat maskTemp = mask.clone();
    cv::floodFill(maskTemp, cv::Point(StartingPointX, StartingPointY), CV_RGB(0, 0, 0));
    ///Inversion du mask
    maskTemp = ~maskTemp;
    mask = maskTemp & mask;
    cv::floodFill(mask, cv::Point(0,0), CV_RGB(255, 255, 255));

    ///On enlève les parasites
    Mat kernel;
    kernel = getStructuringElement(2, Size(7,7), Point(2,2));
    dilate(mask, mask, kernel);
    erode(mask, mask, kernel);

    circle(mask, Point2i(StartingPointX,StartingPointY), 5, Scalar(150,150,150));
    ///Affichage du mask
//    namedWindow("mask3",WINDOW_AUTOSIZE);
//    imshow("mask3", mask);

    ///On retourne le mask
    return mask;

}

/// Fonction permettant de récupérer les 4 coins du plan
vector<Point2i> EdgeDetection::getCorner(Mat img) {

    ///Initialisation des variables
    Mat mask = colorCalibration(img);
    std::vector<KeyPoint> keypoints;
    vector<Point2i> coordCorner;
    ///Déclaration et calcul de l'image hsv
    Mat hsv;
    cvtColor(img, hsv, CV_BGR2HSV);

    ///Paramètre pour la détection des composantes connexes
    SimpleBlobDetector::Params params;
    params.minThreshold = 0;
    params.maxThreshold = 100;
    params.filterByArea = true;
    params.minArea = 100;
    params.maxArea = 10000;
    params.filterByCircularity = false;
    params.filterByConvexity = false;
    params.filterByInertia = false;

    Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    detector->detect(mask, keypoints);

//    drawKeypoints( mask, keypoints, mask, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    /// si plus de 4 composantes connexes trouvées on prends les 4 plus grosses
    if(keypoints.size() > 3 && keypoints.size() == 6){
        ///On cherche le plus grand et on le prends
        for(int i=0 ; i<4 ; i++ ){
            int imax = 0;
            for(int j = 1 ; j< keypoints.size() ; j++) {
                if (keypoints[j].size > keypoints[imax].size) {
                    imax = j;
                }
            }
            coordCorner.push_back(keypoints[imax].pt);
            keypoints[imax].size = 0;
        }
    }


    /// Si on a 4 coins, on met à jour le point de calibration
    if(coordCorner.size() == 4) {
        coordCorner = sortPoints(coordCorner, hsv);
        StartingPointX = (coordCorner[0].x + coordCorner[1].x)/2 ;
        StartingPointY = (coordCorner[0].y + coordCorner[1].y)/2 ;
    }

    return coordCorner;

}

///Fonctions permettant de détecter le départ et l'arrivé de la boule
vector<Point2i> EdgeDetection::startEndDetection(Mat img) {

    ///Initialisation des variables
    Mat mask = colorCalibration(img);
    std::vector<KeyPoint> point;
    vector<Point2i> coordPoint;

    ///Paramètre pour la détection des composantes connexes
    SimpleBlobDetector::Params params;
    params.minThreshold = 0;
    params.maxThreshold = 100;
    params.filterByArea = true;
    params.minArea = 100;
    params.maxArea = 10000;
    params.filterByCircularity = false;
    params.filterByConvexity = false;
    params.filterByInertia = false;

    Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    detector->detect(mask, point);

    /// si plus de 2 composantes connexes trouvées on prends les 2 plus petites
    if(point.size() > 1 && point.size() == 6){
        for(int i=0 ; i<2 ; i++ ){
            int imin = 0;
            for(int j = 1 ; j< point.size() ; j++) {
                if (point[j].size < point[imin].size) {
                    imin = j;
                }
            }
            coordPoint.push_back(point[imin].pt);
            point[imin].size = 2000;
        }
    }

    return coordPoint;

}

/// fonction utilisée pour trier les points
bool sortByY(Point p1, Point p2){
    return p1.y>p2.y ;
}

vector<Point2i> EdgeDetection::sortPoints(vector<Point2i> coord, Mat imgHSV){
    /// tri du y le plus grand au plus petit
    sort(coord.begin(), coord.end(), sortByY);
    /// comparaison des deux du bas et des deux du haut
    if(coord[0].x > coord[1].x) swap(coord[0],coord[1]);
    if(coord[2].x > coord[3].x) swap(coord[2],coord[3]);
    ///réarangement qui marche =)
    swap(coord[1],coord[2]);
    swap(coord[2],coord[3]);


    int compt =0;// sécurité
    ///tant que le point unique n'est pas en premier on fait une rotation
    while((int)imgHSV.at<Vec3b>(coord[0].y,coord[0].x)[0] <= 125 && (compt < 4)){
        std::rotate(coord.begin(),coord.begin()+1,coord.end());
        compt++;
    }


    return coord;
}






///Fonction permettant la détection des lignes
vector<vector<Point2i>> EdgeDetection::linesDetection(Mat img, vector<Point2i> coordCorner){
    /// détection des contours avec Canny
    Mat imgCanny;

    cvtColor(img, imgCanny, COLOR_RGB2GRAY);
    blur( imgCanny, imgCanny, Size(3,3) );
    Canny(imgCanny, imgCanny, 50, 200, 3);

    /// detection des lignes dans le vect lines
    /// vecteur dans lequel sont stockées les lignes
    ///     lignes stockées sous la forme (x1,y1,x2,y2)
    vector<Vec4i> lines;
    /// houghLinesP(imgsource,
    /// vectdest,
    /// distance resolution en pixels
    /// angle resolution en rad
    /// seuil :The minimum number of intersections to “detect” a line
    /// longueur min d'une ligne détectée
    /// max ecart entre pixels de la ligne)

    HoughLinesP(imgCanny, lines, 1, CV_PI/180, 30, 15, 10);

    /// tableau de couples de points
    vector<vector<Point2i>> vectLines;

    ///Initialisation du mask
    Mat mask = Mat::zeros(img.size(), CV_8UC1);

    ///Si on a 4 points alors
    ///On déssine un polygone avec ces 4 points dans le mask
    if(coordCorner.size() == 4) {
        ///Conversion des données pour utiliser la fonction fillPoly
        Point coord[1][4];
        coord[0][0] = coordCorner[0];
        coord[0][1] = coordCorner[1];
        coord[0][2] = coordCorner[2];
        coord[0][3] = coordCorner[3];
        ///Nombre de points
        int npt[] = {4};
        ///Pointeur de points
        const Point *ppt[1] = {coord[0]};

        fillPoly(mask, ppt, npt, 1, Scalar(255, 255, 255), 8);
    }

    for(Vec4i l : lines){

        /// couple de points
        vector<Point2i> vectPoints ;
        vectPoints.emplace_back(l[0], l[1]);
        vectPoints.emplace_back(l[2], l[3]);



        ///tracé de la ligne
        if((int)mask.at<uchar>(vectPoints[0].y, vectPoints[0].x) == 255 && (int)mask.at<uchar>(vectPoints[1].y, vectPoints[1].x) == 255) {
            /// ajout du couple au tableau
            vectLines.push_back(vectPoints) ;
            //line( imgCanny, vectPoints[0], vectPoints[1], Scalar(0,0,0), 1, CV_AA);
        }
    }

//    namedWindow("canny",WINDOW_AUTOSIZE);
//    imshow("canny", imgCanny);

    return(filterDouble(vectLines,10));
    //return(vectLines);
}


///Fonction permettant de trier les murs
vector<vector<Point2i>> EdgeDetection::filterDouble(vector<vector<Point2i>> vectLines, int thresh){
    vector<vector<Point2i>> linesFilter = vectLines ;
    int suppressed = 0;
    int incr = 0 ;
    for(vector<Point2i> line : vectLines) {
        bool isGood = true;
        for (vector<Point2i> goodLine : linesFilter) {
            if (line[0].x != goodLine[0].x && line[0].y != goodLine[0].y && line[1].x != goodLine[1].x &&
                line[1].y != goodLine[1].y) {
                int minX = min(goodLine[0].x, goodLine[1].x) - thresh;
                int maxX = max(goodLine[0].x, goodLine[1].x) + thresh;
                int minY = min(goodLine[0].y, goodLine[1].y) - thresh;
                int maxY = max(goodLine[0].y, goodLine[1].y) + thresh;
                isGood &= (line[0].x > maxX || line[0].x < minX || line[0].y > maxY || line[0].y < minY) ||
                          (line[1].x > maxX || line[1].x < minX || line[1].y > maxY || line[1].y < minY);
            }
        }
        if (!isGood) {
            linesFilter.erase(linesFilter.begin()+ (incr-suppressed));
            suppressed ++;
        }
        incr++ ;
    }

    return linesFilter ;
}


///Fonction permettant de savoir s'il y a une inversion dans le plan
bool EdgeDetection::isReversed(vector<Point2i> &corners){
    vector<Point2i> cop;
    for(int i = 0; i < corners.size(); i++){
        cop.push_back(corners[i]);
    }
    sort(cop.begin(), cop.end(), sortByY);
    return fabs(cop[0].x - cop[1].x) < fabs(cop[2].x - cop[3].x);
}

