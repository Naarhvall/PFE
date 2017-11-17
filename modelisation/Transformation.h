#ifndef AMAZED_TRANSFORMATION_H
#define AMAZED_TRANSFORMATION_H

#include <opencv2/core/core.hpp>

class Transformation {

private:

    const float FX =1200;/// distance focale x de la caméra
    const float FY =1200;/// distance focale y de la caméra
    double X0;          /// centre de l'image en x
    double Y0;          /// centre de l'image en y

    cv::Size size;      /// size.width => w, size.height => h
    double N;           /// near
    double F;           /// far

    cv::Mat K;          /// Intrinsic
    cv::Mat P;          /// Extrinsic
    cv::Mat H;          /// Homography
    cv::Mat NDC;        /// NDC
    cv::Mat Persp;      /// Persp
    cv::Mat rotMatrix;  /// Matrice 3x3 de rotation

    cv::Mat proj;       /// proj
    cv::Mat modelView;  /// modelview
    
    /** Calcule la matrice d'homographie à l'aide de la fonction openCV findHomography
     * @params
     *      edgeCoordinate : coordonnés des quatre coins sur le flux caméra
     */
    void computeHomographyMatrix(const std::vector<cv::Point2i> &edgeCoordinate);
    
    /** Calcule la matrice intrinseque
     */
    void computeIntrinsicMatrix();
    
    /** Calcule la matrice Extrinsèque
     * @params
     *      edgeCoordinate : coordonnés des quatre coins sur le flux caméra
     */
    void computeExtrinsicMatrix(const std::vector<cv::Point2i> &edgeCoordinate);
    
    /** Calcule la matrice de projection en multipliant la matrice NDC et Persp
     */
    void computeProjMatrix();
    
    /** Calcule la matrice ModelView
     * @params
     *      edgeCoordinate : coordonnés des quatre coins sur le flux caméra
     */
    void computeModelviewMatrix(const std::vector<cv::Point2i> &edgeCoordinate);
    
    /** Calcule de a matrice NDC (Normalized Device Coordinate) (équivalente à celle donnée apar glOrtho)
     */
    void computeNDCMatrix();
    
    /** Calcule de la matrice de perspective
     */
    void computePerspMatrix();

public:
    
    /** Constructeur
     *      edgeCoordinate : coordonnés des quatre coins sur le flux caméra
     *      size : taille de l'écran
     *      near : distance minimale a afficher devant la caméra
     *      far : distance maximale a afficher devant la caméra
     */
    Transformation(std::vector<cv::Point2i> &edgeCoordinate, cv::Size size, double near, double far);

    /** Constructeur par défaut
     */
    Transformation();

    /** Permet de créer une matrice compatible avec openGL de modelView
     */
    void getModelviewMatrix(double matrix[]);
    
    /** Permet de créer une matrice compatible avec openGL de projection
     */
    void getProjectionMatrix(double matrix[]);
    
    /** @return
     *      La matrice d'homographie 3x3
     */
    cv::Mat getHomography();

    /**
     * Calcul les angles d'Euler de la rotation appliqué
     * @return
     *      Un vecteur contenant les 3 angles de rotation autour de x, y, z
     */
    cv::Vec3d getEulerAngle();

    /**
     * Calcule le point du modele associé au point image de l'image en parametre
     * @param imagePoint : le point image en coordonées pixel repère openCV
     * @return
     *      Le point model correspondant au point image
     */
    cv::Point2d getModelPointFromImagePoint(const cv::Point2d &imagePoint) const;
};


#endif //AMAZED_TRANSFORMATION_H
