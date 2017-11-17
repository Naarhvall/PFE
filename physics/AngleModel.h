#ifndef AMAZED_ANGLE_H
#define AMAZED_ANGLE_H


#include "../modelisation/Transformation.h"

class AngleModel {

private:

    Transformation *initialTransformation;  /// Objet Transformation de la transformation initiale
    Transformation *currentTransformation;  /// Objet de Transformation de la transformation courante

    double angleX;  /// Angle autour de l'axe X entre les 2 transformations
    double angleY;  /// Angle autour de l'axe Y entre les 2 transformations
    double angleZ;  /// Angle autour de l'axe Z entre les 2 transformations

    /**
     * Calcule le produit scalaire entre les 2 vecteurs
     * @param a : premier vecteur
     * @param b : deuxieme vecteur
     * @return Le produit scalaire entre les 2 vecteurs
     */
    double scalarProduct(cv::Mat const &a, cv::Mat const &b) const;

    /**
     * Calcule la norme du produit vectorielle entre les 2 vecteurs
     * @param a : premier vecteur
     * @param b : deuxieme vecteur
     * @return La norme du produit vectorielle entre les 2 vecteurs
     */
    double crossProduct(cv::Mat const &a, cv::Mat const &b) const;

    /**
     * Calcule la norme d'un vecteur
     * @param a : le vecteur
     * @return La norme du vecteur
     */
    double norm(cv::Mat const &a)const ;

    /**
     * Calcule l'angle oriente entre les 2 vecteurs
     * @param a : premier vecteur
     * @param b : deuxieme vecteur
     * @return La valeur de l'angle en radian dans l'intervalle [-pi; pi]
     */
    double angleOriente(cv::Mat &a, cv::Mat &b) const;

    /**
     * Calcule les angles entre les 2 transformations courantes et les stocke en tant qu'attribut
     */
    void computeAngle();

public:

    /**
     * Définie la transformation courante
     * @param currentTransformation pointeur sur la transformation courante
     */
    void setCurrentTransformation(Transformation *currentTransformation);

    /**
     * Constructeur avec la transformation initiale
     * @param transformation : la transformation initiale
     */
    explicit AngleModel(Transformation *transformation);

    /**
     * @return renvoie la valeur de l'angle autour de l'axe X entre les deux transformations
     */
    double getAngleX() const;

    /**
     * @return renvoie la valeur de l'angle autour de l'axe Y entre les deux transformations
     */
    double getAngleY() const;

    /**
     * @return renvoie la valeur de l'angle autour de l'axe Z entre les deux transformations
     */
    double getAngleZ() const;

};


#endif //AMAZED_ANGLE_H
