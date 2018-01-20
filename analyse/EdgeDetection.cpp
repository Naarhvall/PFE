#include <iostream>
#include "EdgeDetection.h"
#include "opencv2/opencv.hpp"
#include "../stream/CameraStream.h"
#include <math.h>

using namespace std;
using namespace cv;

///Coordonnées du point permettant la calibration pour la mask sur le niveau de gris
int StartingPointX = 0 ;
int StartingPointY = 0;
int points[4][2];

EdgeDetection::EdgeDetection(Mat img, bool ini){

    if(ini){
        points[0][0] = 0;
        points[0][1] = 0;

        points[1][0] = img.cols;
        points[1][1] = 0;

        points[2][0] = img.cols;
        points[2][1] = img.rows;

        points[3][0] = 0;
        points[3][1] = img.rows;
    }
}

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
    int nivMax = 0;
    int nivGrey, tempx, tempy;

    ///On récupère le niveau de gris max du voisinage du pixel du mileu
    for(int i = -10; i < 10; i++){
      for(int j = -10; j < 10; j++) {
          if (0 <= StartingPointY + i && StartingPointY + i < img.rows && 0 <= StartingPointX + j && StartingPointX + j < img.cols){
              nivGrey = (int) imgGrey.at<uchar>(StartingPointY + i, StartingPointX + j);
              if (nivGrey > nivMax) {
                  nivMax = nivGrey;
                  tempx = StartingPointX + j;
                  tempy = StartingPointY + i;
              }
          }
      }
    }

    StartingPointX = tempx;
    StartingPointY = tempy;

    ///On créé le mask en fonction du niveau de gris précédent
    inRange(imgGrey, nivMax-30, nivMax+60, mask);
    Mat ouais = mask.clone();
    circle(ouais, Point2i(StartingPointX,StartingPointY), 5, Scalar(150,150,150));
    ///Affichage du mask
    namedWindow("mask3",WINDOW_AUTOSIZE);
    imshow("mask3", ouais);

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

    ///On retourne le mask
    return mask;

}

/// Fonction permettant de récupérer les 4 coins du plan
vector<Point2i> EdgeDetection::getCorner(Mat img) {

    ///Initialisation des variables
    Mat mask = colorCalibration(img);
    std::vector<KeyPoint> keypoints;
    vector<Point2i> coordCorner;

    /// cherche les coins sur le masque nouvelle version : pas de blobs
    int rows = mask.rows;
    int cols = mask.cols;
//    int minX = cols , minXy, minY = rows, minYx, maxX =0 , maxXy, maxY=0, maxYx;

    /// recréation d'un mask quivabien
    Mat newMask;
    Mat imgGrey;
    cvtColor(img, imgGrey, COLOR_RGB2GRAY);
    auto midGrey = (int)imgGrey.at<uchar>(StartingPointY, StartingPointX);
    inRange(imgGrey, midGrey-40, midGrey+40, newMask);
    Mat maskTemp = newMask.clone();
    cv::floodFill(maskTemp, cv::Point(StartingPointX, StartingPointY), CV_RGB(0, 0, 0));
    maskTemp = ~maskTemp;
    newMask = maskTemp & newMask;
    maskTemp = newMask.clone();
    cv::floodFill(maskTemp, cv::Point(0,0), CV_RGB(255, 255, 255));
    maskTemp = ~maskTemp;
    newMask= newMask | maskTemp;
    Mat kernel1;
    kernel1 = getStructuringElement(2, Size(7,7), Point(2,2));
    dilate(newMask, newMask, kernel1);
    erode(newMask, newMask, kernel1);

    int kernel = 7;
    int voisin = 30 ;
    int tab[4][3] = {{0,0,0},
                     {0,0,0},
                     {0,0,0},
                     {0,0,0}};

    int nbNoirs,rangProche,minNoirs,rangMin ;
    for(int i=0 ; i<rows ; i++){
        for(int j =0 ; j<cols ; j++){
            if(newMask.at<uchar>(i,j) == 255){
                 nbNoirs = 0;
                ///calcul du nb de noirs dans le voisinage
                for(int ii = i-kernel ; ii <i+kernel ; ii++){
                    for(int jj = j-kernel ; jj<j+kernel ; jj++){
                        if(ii >= 0 && ii<rows && jj>=0 && jj<cols && newMask.at<uchar>(ii,jj) == 0){
                            nbNoirs++;
                        }
                    }
                }
                if(nbNoirs > kernel*kernel*2){
                    ///on regarde si il est proche d'un autre point
                    rangProche = -1 ;
                    for(int iTab = 0 ; iTab<4 ; iTab++){
                        if( i > tab[iTab][2]-voisin && i < tab[iTab][2]+voisin && j > tab[iTab][1]-voisin && j < tab[iTab][1]+voisin){
                            rangProche = iTab ;
                        }
                    }
                    ///si oui
                    if(rangProche != -1 && nbNoirs > tab[rangProche][0]){
                        tab[rangProche][0] = nbNoirs ;
                        tab[rangProche][1] = j ;
                        tab[rangProche][2] = i ;
                    }
                    else if(rangProche == -1){
                        minNoirs = kernel*kernel*4;
                        rangMin = 0;
                        for(int iTab =0 ; iTab<4 ; iTab++){
                            if(tab[iTab][0] < minNoirs){
                                minNoirs = tab[iTab][0];
                                rangMin = iTab ;
                            }
                        }
                        if( minNoirs<nbNoirs){
                            tab[rangMin][0] = nbNoirs ;
                            tab[rangMin][1] = j ;
                            tab[rangMin][2] = i ;
                        }
                    }
                }
            }
        }
    }
    for(int i=0 ; i<4 ; i++){
        coordCorner.push_back(Point(tab[i][1],tab[i][2]));
    }

    /// On ordonne nos points
    coordCorner = pointsVerification(coordCorner);

    for(int i = 0; i < 4; i++){
        points[i][0] = coordCorner[i].x;
        points[i][1] = coordCorner[i].y;
        circle(newMask, Point(coordCorner[i].x,coordCorner[i].y),i*20+8,Scalar(255,0,0));
    }

    StartingPointX = (coordCorner[0].x + coordCorner[1].x + coordCorner[2].x + coordCorner[3].x)/4 ;
    StartingPointY = (coordCorner[0].y + coordCorner[1].y + coordCorner[2].y + coordCorner[3].y)/4 ;
    namedWindow("mask",WINDOW_AUTOSIZE);
    imshow("mask", newMask);
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
    /// si on a les 2 composantes connexes
    if(point.size() == 2){
        for(int i=0 ; i<2 ; i++ ){
            coordPoint.push_back(point[i].pt);
        }
    }
    return coordPoint;
}

/// fonction utilisée pour trier les points
bool sortByY(Point p1, Point p2){
    return p1.y>p2.y ;
}


vector<Point2i> EdgeDetection::pointsVerification(vector<Point2i> coord){
    vector<Point2i> temp;
    int valTemp1 = 10000, valTemp2, indice;

    /// On cherche min x+y
    for(int i = 0; i < 4; i++){
        valTemp2 = coord[i].x+coord[i].y;
        if(valTemp2 < valTemp1) {
            valTemp1 = valTemp2;
            indice = i;
        }
    }
    temp.push_back(coord[indice]);

    /// On cherche max x-y
    valTemp1 = 0;
    for(int i = 0; i < 4; i++){
        valTemp2 = coord[i].x-coord[i].y;
        if(valTemp2 > valTemp1) {
            valTemp1 = valTemp2;
            indice = i;
        }
    }
    temp.push_back(coord[indice]);

    ///On cherche max x+y
    valTemp1 = 0;
    for(int i = 0; i < 4; i++){
        valTemp2 = coord[i].x+coord[i].y;
        if(valTemp2 > valTemp1) {
            valTemp1 = valTemp2;
            indice = i;
        }
    }
    temp.push_back(coord[indice]);

    ///On cherche max y-x
    valTemp1 = 0;
    for(int i = 0; i < 4; i++){
        valTemp2 = coord[i].y-coord[i].x;
        if(valTemp2 > valTemp1) {
            valTemp1 = valTemp2;
            indice = i;
        }
    }
    temp.push_back(coord[indice]);

    return temp;
}

///Fonction permettant la détection des lignes
vector<vector<Point2i>> EdgeDetection::wallsDetection(Mat img, vector<Point2i> coordCorner, vector<Point2i> coordStartEnd){
    /// détection des contours avec Canny

    /// detection des lignes dans le vect lines
    /// vecteur dans lequel sont stockées les lignes
    ///     lignes stockées sous la forme (x1,y1,x2,y2)
    /// houghLinesP(imgsource,
    /// vectdest,
    /// distance resolution en pixels
    /// angle resolution en rad
    /// seuil :The minimum number of intersections to “detect” a line
    /// longueur min d'une ligne détectée
    /// max ecart entre pixels de la ligne)

    Mat newMask;
    Mat imgGrey;
    cvtColor(img, imgGrey, COLOR_RGB2GRAY);
    auto midGrey = (int)imgGrey.at<uchar>(StartingPointY, StartingPointX);
    inRange(imgGrey, midGrey-40, midGrey+40, newMask);
    Mat maskTemp = newMask.clone();
    cv::floodFill(maskTemp, cv::Point(StartingPointX, StartingPointY), CV_RGB(0, 0, 0));
    maskTemp = ~maskTemp;
    newMask = maskTemp & newMask;
    maskTemp = newMask.clone();
    cv::floodFill(maskTemp, cv::Point(0,0), CV_RGB(255, 255, 255));
    maskTemp = ~maskTemp;

    if(coordStartEnd.size() == 2){
        floodFill(maskTemp, coordStartEnd[0],Scalar(0,0,0));
        floodFill(maskTemp,coordStartEnd[1],Scalar(0,0,0));
    }

//    namedWindow("maskMurs",WINDOW_AUTOSIZE);
//    imshow("maskMurs", maskTemp);

    /// tableau de couples de points
    vector<Vec4i> vectLinesTemp;
    vectLinesTemp = linesDetection(maskTemp, 25,25,5);

    vector<vector<Point2i>> vectLines;
    vector<Point2i> lineTemp ;
//    lineTemp.push_back(coordCorner[0]);
//    lineTemp.push_back(coordCorner[1]);
//    vectLines.push_back(lineTemp);
//    lineTemp.clear();
//    lineTemp.push_back(coordCorner[1]);
//    lineTemp.push_back(coordCorner[2]);
//    vectLines.push_back(lineTemp);
//    lineTemp.clear();
//    lineTemp.push_back(coordCorner[2]);
//    lineTemp.push_back(coordCorner[3]);
//    vectLines.push_back(lineTemp);
//    lineTemp.clear();
//    lineTemp.push_back(coordCorner[3]);
//    lineTemp.push_back(coordCorner[0]);
//    vectLines.push_back(lineTemp);
//    lineTemp.clear();

    for(Vec4i line: vectLinesTemp){
        cv::line(maskTemp, Point(line[0],line[1]),Point(line[2],line[3]),Scalar(0,0,0),3, CV_AA);
        lineTemp.push_back(Point(line[0],line[1]));
        lineTemp.push_back(Point(line[2],line[3]));
        vectLines.push_back(lineTemp);
        lineTemp.clear();
    }
    vectLinesTemp.clear();

    int nbWhite = 0;
    for(int i = 0 ; i<maskTemp.cols ; i++){
        for(int j = 0 ; j<maskTemp.rows ; j++){
            if((int)maskTemp.at<uchar>(j,i) == 255)
                nbWhite ++ ;
        }
    }

//    namedWindow("maskMursApres1erePasse",WINDOW_AUTOSIZE);
//    imshow("maskMursApres1erePasse", maskTemp);

    if(nbWhite > 0){
        vectLinesTemp = linesDetection(maskTemp,20,10,3);
        for(Vec4i line: vectLinesTemp){
            cv::line(maskTemp, Point(line[0],line[1]),Point(line[2],line[3]),Scalar(0,0,0),3, CV_AA);
            lineTemp.push_back(Point(line[0],line[1]));
            lineTemp.push_back(Point(line[2],line[3]));
            vectLines.push_back(lineTemp);
            lineTemp.clear();
        }
    }

//    namedWindow("maskMursApres2emePasse",WINDOW_AUTOSIZE);
//    imshow("maskMursApres2emePasse", maskTemp);

    return(filterDouble(vectLines,15));
}

vector<Vec4i> EdgeDetection::linesDetection(Mat mask, int thresh, int minLength, int maxGap ){
    vector<Vec4i> linesVect ;
    HoughLinesP(mask, linesVect, 1, CV_PI/180, thresh, minLength, maxGap);
    return linesVect ;
}

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
//    bool doNext = true ;
//    vector<vector<Point2i>> linesTemp ;
//    vector<Point2i> temp;
//    vector<bool> isUsed = vector<bool>(linesFilter.size(),false);
//    float vect1,vect2;
//    float threshVect = 0.07;
//    bool noMatch;
//    cout << linesFilter.size() << endl ;
//    while(doNext ==1){
//        doNext = false ;
//        //parcours de tout les murs
//        for(int currentWall = 0 ; currentWall < linesFilter.size() ; currentWall++){
//            //si mur non deja utilisé
//            noMatch = true ;
//            if(!isUsed[currentWall]) {
//                //calcul de son vect directeur
//                if (linesFilter[currentWall][0].y != linesFilter[currentWall][1].y)
//                    vect1 = (linesFilter[currentWall][0].x - linesFilter[currentWall][1].x) /
//                            (linesFilter[currentWall][0].y - linesFilter[currentWall][1].y);
//                else vect1 = -1;
//                //parcours des autres murs
//                for (int otherWall = 0; otherWall < linesFilter.size(); otherWall++) {
//                    //si different et pas deja utilise
//                    if (otherWall != currentWall && !isUsed[otherWall]) {
//                        // calcul du 2eme vect dir
//                        if (linesFilter[otherWall][0].y != linesFilter[otherWall][1].y)
//                            vect2 = (linesFilter[otherWall][0].x - linesFilter[otherWall][1].x) /
//                                    (linesFilter[otherWall][0].y - linesFilter[otherWall][1].y);
//                        else vect2 = -1;
//                        // si mm vects dirs
//                        if (vect2 < vect1 + threshVect && vect2 > vect1 - threshVect) {
//                            //si ~2 mm murs
//                            if(linesFilter[otherWall][0].x < linesFilter[currentWall][0].x+thresh &&
//                               linesFilter[otherWall][0].x > linesFilter[currentWall][0].x-thresh &&
//                               linesFilter[otherWall][0].y < linesFilter[currentWall][0].y+thresh &&
//                               linesFilter[otherWall][0].y > linesFilter[currentWall][0].y-thresh &&
//                               linesFilter[otherWall][1].x < linesFilter[currentWall][1].x+thresh &&
//                               linesFilter[otherWall][1].x > linesFilter[currentWall][1].x-thresh &&
//                               linesFilter[otherWall][1].y < linesFilter[currentWall][1].y+thresh &&
//                               linesFilter[otherWall][1].y > linesFilter[currentWall][1].y-thresh ||
//
//                               linesFilter[otherWall][0].x < linesFilter[currentWall][1].x+thresh &&
//                               linesFilter[otherWall][0].x > linesFilter[currentWall][1].x-thresh &&
//                               linesFilter[otherWall][0].y < linesFilter[currentWall][1].y+thresh &&
//                               linesFilter[otherWall][0].y > linesFilter[currentWall][1].y-thresh &&
//                               linesFilter[otherWall][1].x < linesFilter[currentWall][0].x+thresh &&
//                               linesFilter[otherWall][1].x > linesFilter[currentWall][0].x-thresh &&
//                               linesFilter[otherWall][1].y < linesFilter[currentWall][0].y+thresh &&
//                               linesFilter[otherWall][1].y > linesFilter[currentWall][0].y-thresh){
//                                temp.push_back(Point2i(linesFilter[currentWall][0].x,linesFilter[currentWall][0].y));
//                                temp.push_back(Point2i(linesFilter[currentWall][1].x,linesFilter[currentWall][1].y));
//                                linesTemp.push_back(temp);
//                                temp.clear();
//                                doNext = true;
//                                isUsed[currentWall] = true ;
//                                isUsed[otherWall] = true ;
//                                noMatch = false;
//                            }
//                            else if(linesFilter[otherWall][0].x < linesFilter[currentWall][0].x+thresh &&
//                                    linesFilter[otherWall][0].x > linesFilter[currentWall][0].x-thresh &&
//                                    linesFilter[otherWall][0].y < linesFilter[currentWall][0].y+thresh &&
//                                    linesFilter[otherWall][0].y > linesFilter[currentWall][0].y-thresh ){
//                                temp.push_back(Point2i(linesFilter[currentWall][1].x,linesFilter[currentWall][1].y));
//                                temp.push_back(Point2i(linesFilter[otherWall][1].x,linesFilter[otherWall][1].y));
//                                linesTemp.push_back(temp);
//                                temp.clear();
//                                doNext = true;
//                                isUsed[currentWall] = true ;
//                                isUsed[otherWall] = true ;
//                                noMatch = false;
//                            }
//                            else if(linesFilter[otherWall][1].x < linesFilter[currentWall][1].x+thresh &&
//                                    linesFilter[otherWall][1].x > linesFilter[currentWall][1].x-thresh &&
//                                    linesFilter[otherWall][1].y < linesFilter[currentWall][1].y+thresh &&
//                                    linesFilter[otherWall][1].y > linesFilter[currentWall][1].y-thresh){
//                                temp.push_back(Point2i(linesFilter[currentWall][0].x,linesFilter[currentWall][0].y));
//                                temp.push_back(Point2i(linesFilter[otherWall][0].x,linesFilter[otherWall][0].y));
//                                linesTemp.push_back(temp);
//                                temp.clear();
//                                doNext = true;
//                                isUsed[currentWall] = true ;
//                                isUsed[otherWall] = true ;
//                                noMatch = false;
//                            }
//                            else if(linesFilter[otherWall][0].x < linesFilter[currentWall][1].x+thresh &&
//                                    linesFilter[otherWall][0].x > linesFilter[currentWall][1].x-thresh &&
//                                    linesFilter[otherWall][0].y < linesFilter[currentWall][1].y+thresh &&
//                                    linesFilter[otherWall][0].y > linesFilter[currentWall][1].y-thresh){
//                                temp.push_back(Point2i(linesFilter[currentWall][0].x,linesFilter[currentWall][0].y));
//                                temp.push_back(Point2i(linesFilter[otherWall][1].x,linesFilter[otherWall][1].y));
//                                linesTemp.push_back(temp);
//                                temp.clear();
//                                doNext = true;
//                                isUsed[currentWall] = true ;
//                                isUsed[otherWall] = true ;
//                                noMatch = false;
//                            }
//                            else if(linesFilter[otherWall][1].x < linesFilter[currentWall][0].x+thresh &&
//                                    linesFilter[otherWall][1].x > linesFilter[currentWall][0].x-thresh &&
//                                    linesFilter[otherWall][1].y < linesFilter[currentWall][0].y+thresh &&
//                                    linesFilter[otherWall][1].y > linesFilter[currentWall][0].y-thresh){
//                                temp.push_back(Point2i(linesFilter[currentWall][1].x,linesFilter[currentWall][1].y));
//                                temp.push_back(Point2i(linesFilter[otherWall][0].x,linesFilter[otherWall][0].y));
//                                linesTemp.push_back(temp);
//                                temp.clear();
//                                doNext = true;
//                                isUsed[currentWall] = true ;
//                                isUsed[otherWall] = true ;
//                                noMatch = false;
//                            }
//                        }
//                    }
//                }
//                if (noMatch){
//                    temp.push_back(Point2i(linesFilter[currentWall][1].x,linesFilter[currentWall][1].y));
//                    temp.push_back(Point2i(linesFilter[currentWall][0].x,linesFilter[currentWall][0].y));
//                    linesTemp.push_back(temp);
//                    temp.clear();
//                }
//            }
//        }
//        //linesTemp : nouveaux murs
//        //linesFilter : anciens murs
//        //isUsed : anciens murs utilisés
//        linesFilter = linesTemp;
//        linesTemp.clear();
//        isUsed = vector<bool>(linesFilter.size(),false);
//        cout << linesFilter.size() << endl ;
//    }
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

