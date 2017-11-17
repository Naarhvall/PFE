
#ifndef AMAZED_VECTOR2D_H
#define AMAZED_VECTOR2D_H


class Vector2d {
    private:
        double _x1, _y1, _x2, _y2;
        double _X, _Y;                  // pour les calculs

    public:
        /// constructeurs
        Vector2d(double x1, double y1, double x2, double y2);
        Vector2d(double X, double Y);

        /// Projette le vecteur sur v2
        double projectionOn(Vector2d v2);

        /// "Oriente" le vecteur de gauche a droite, bas en haut
        void sortPoints();

        void setMagnitude(double magnitude);

        ///Calcule la norme d'un vecteur
        double getMagnitude();

        /**
        * Method to obtain vector unit of current vector
        * @return A copy of normalised vector
        */
        Vector2d normalize();

        /**
         *  Les vecteurs doivent etre de la meme taille
         * @param vect
         * @return Le produit vectoriel des 2 vecteurs
         */
        double dotProduct(Vector2d vect);

        /**
         *  Change les composantes X et Y du vecteur
         * @param magnitude
         * @param angleRadian
         */
        void polar(double magnitude, double angleRadian);

        double get_x1() const;

        void set_x1(double _x1);

        double get_y1() const;

        void set_y1(double _y1);

        double get_x2() const;

        void set_x2(double _x2);

        double get_y2() const;

        void set_y2(double _y2);

        double get_X() const;

        void set_X(double _X);

        double get_Y() const;

        void set_Y(double _Y);
};


#endif //AMAZED_VECTOR2D_H
