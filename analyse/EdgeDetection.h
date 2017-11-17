
#ifndef PROJET_AMAZED_EDGEDETECTION_H
#define PROJET_AMAZED_EDGEDETECTION_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "../stream/CameraStream.h"

class EdgeDetection {

private:

    /** Trie le vector de 4 points comme suit:
     *          le point particulier (couleur différente) en premier et les suivants dans le sens horaire
     *
     * @param coord le vector de coordonnées des quatres coins
     * @param imgHSV la frame courante en hsv
     * @return retourne le vector de 4 points trié
     */
    std::vector<cv::Point2i> sortPoints(std::vector<cv::Point2i> coord, cv::Mat imgHSV);

    /** enlève les doublons de lignes
     *
     * @param vectLines tableau de lignes à filtrer
     * @param thresh seuil de proximité des lignes
     * @return return le tableau de ligne sans doublons
     */
    std::vector<std::vector<cv::Point2i>> filterDouble(std::vector<std::vector<cv::Point2i>> vectLines, int thresh);

public:

     /** Permet de trouver la teinte et la saturation des coins du plan
      * @param
     *      img : l'image où trouver les lignes
     * @return
     *      renvoie un mask du plan
     */
    cv::Mat colorCalibration(cv::Mat img);

    /** détecte les couleurs d'une image
     * @params
     *      img : l'image où il faut détecter les couleurs
     */
    std::vector<cv::Point2i> getCorner(cv::Mat img);

    /** détecte les lignes dans une image et renvoie les coordonnées des extrémités
     * @params
     *      img : l'image où trouver les lignes
     *      coordCorner : coin du plan
     * @return
     *      renvoie un vecteur de lignes
     *      les lignes sont de la forme Point(x1,y1) Point(x2,y2) qui sont les coordonnées des extrémités d'une ligne
     * */
    std::vector<std::vector<cv::Point2i>> linesDetection(cv::Mat img, std::vector<cv::Point2i> coordCorner);

    /** détecte les points d'arrivé et de départ
     * @param
     *      img : l'image où trouver les lignes
     * @return
     *      renvoie les coordonnées des deux points
     */
    std::vector<cv::Point2i> startEndDetection(cv::Mat img);

    /**
     *
     * @param corners
     * @return
     */
    bool isReversed(std::vector<cv::Point2i> &corners);
};


#endif //PROJET_AMAZED_EDGEDETECTION_H
