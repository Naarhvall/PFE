
#ifndef PROJET_AMAZED_EDGEDETECTION_H
#define PROJET_AMAZED_EDGEDETECTION_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "../stream/CameraStream.h"

class EdgeDetection {

private:

    /** Trie les 4 points selon les positions des points de la frame d'avant
     *
     * @param coord le vector de coordonnées des quatres coins
     * @return retourne le vector de 4 points trié
     */
    std::vector<cv::Point2i> sortPoints(std::vector<cv::Point2i> coord);

    /** Verifie que les points sont dans le bonne ordre et le remets le cas contraire
    *
    * @param coord le vector de coordonnées des quatres coins
    * @return retourne le vector de 4 points trié
    */
    std::vector<cv::Point2i> pointsVerification (std::vector<cv::Point2i> coord);

    /** enlève les doublons de lignes
     *
     * @param vectLines tableau de lignes à filtrer
     * @param thresh seuil de proximité des lignes
     * @return return le tableau de ligne sans doublons
     */
    std::vector<std::vector<cv::Point2i>> filterDouble(std::vector<std::vector<cv::Point2i>> vectLines, int thresh);

public:

    EdgeDetection(cv::Mat img, bool ini);

     /** Permet de trouver la teinte et la saturation des coins du plan
      * @param
     *      img : l'image où trouver les lignes
     * @return
     *      renvoie un mask du plan
     */
    cv::Mat colorCalibration(cv::Mat img);

    /** contruit le masque qui sert de base a tout les autres
      * @param
     *      img : l'image de depart
     * @return
     *      renvoie le masque cree
     */
    cv::Mat buildBasicMask(cv::Mat img);

    /** détecte les coins de la feuille dans le voisinnage des anciesn coins
     * si les coins ne sont pas convenables, la fonction appelle getCornerMinMax
     * @params
     *      img : l'image où il faut détecter les coins
     */
    std::vector<cv::Point2i> getCorner(cv::Mat img);

    /** détecte les coins de la feuille en parcourant entierement le masque
     * @params
     *      mask : le masque où il faut détecter les coins
     */
    std::vector<cv::Point2i> getCornerMinMax(cv::Mat mask);


    /** détecte les lignes dans une image et renvoie les coordonnées des extrémités
     * @params
     *      img : l'image où trouver les lignes
     *      coordCorner : coin du plan
     * @return
     *      renvoie un vecteur de lignes
     *      les lignes sont de la forme Point(x1,y1) Point(x2,y2) qui sont les coordonnées des extrémités d'une ligne
     * */
    std::vector<std::vector<cv::Point2i>> wallsDetection(cv::Mat img, std::vector<cv::Point2i> coordCorner, std::vector<cv::Point2i> coordStartEnd);

    std::vector<cv::Vec4i> linesDetection(cv::Mat mask, int thresh, int minLength, int maxGap);

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
