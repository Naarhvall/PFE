#include <GL/freeglut.h>
#include <opencv2/core/core.hpp>
#include "OpenGL.h"
#include "../physics/CollisionDetection.h"
#include <ctime>

using namespace cv;
using namespace std;

/// Fonction appelé en boucle et définie dans le main
void loop(int);

OpenGL::OpenGL(GlutMaster * glutMaster, int setWidth, int setHeight, int setInitPositionX, int setInitPositionY, char * title, Ball *ball, CameraStream * cameraStream, bool anaglyph){

    this->ball = ball;
    this->cameraStream = cameraStream;

    this->width  = setWidth;
    this->height = setHeight;

    this->initPositionX = setInitPositionX;
    this->initPositionY = setInitPositionY;

    this->p = new double[16];
    this->m = new double[16];

    this->anaglyph = anaglyph;

    this->textMaze = imread(assetsPath + "assets/mazeGround.png"); ///texture du sol du labyrinthe
    this->textWall = imread(assetsPath + "assets/mazeWall.png"); ///texture du mur du labyrinthe
    this->textFlag = imread(assetsPath + "assets/mazeFlag.png"); ///texture du drapeau d'arrivée

    if(anaglyph){
        this->textCam = imread(assetsPath + "assets/noir.jpg"); /// fond noir
    }

    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE | GLUT_STENCIL);
    glutInitWindowSize(this->width, this->height);
    glutInitWindowPosition(this->initPositionX, this->initPositionY);
    glViewport(0, 0, this->width, this->height);

    glutMaster->CallGlutCreateWindow(title, this);
    glGenTextures(4, textArray);
    loadTexture(textArray[ID_TEXT_MAZE], textMaze);
    loadTexture(textArray[ID_TEXT_WALL], textWall);
    loadTexture(textArray[ID_TEXT_FLAG], textFlag);
    applicateLight();
    glEnable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 	// Nécessaire pour éviter une déformation de l'image
}

OpenGL::~OpenGL(){
    glutDestroyWindow(windowID);
    delete p;
    delete m;
}

void OpenGL::CallBackDisplayFunc(){

    bool endGame = false;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);

    if(!anaglyph){
        this->textCam = cameraStream->getCurrentFrame();
    }else{
        rectangle(this->textCam, Point(0, 0), Point(200, 200), Scalar(0, 0, 0), -1);
    }

    putText(this->textCam, to_string(fps), Point2i(0, 10), FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 0, 255), 1);
    putText(this->textCam, "press q to exit", Point2i(this->textCam.cols - 80, 10), FONT_HERSHEY_PLAIN, 0.6, Scalar(0, 0, 255), 1);

    /// Si la balle atteint la fin on affiche l'écran de fin avec le temps
    if(CollisionDetection::hasArrived(ball, this->getEndPoint())){
        destroyAllWindows();
        this->textCam = imread(assetsPath + "assets/mazeEnd.png");
        putText(this->textCam, to_string( (int) difftime( time(nullptr), start)) + "s", Point2i(530, 492), FONT_HERSHEY_PLAIN, 4, Scalar(225, 238, 251), 4);
        endGame = true;
    }

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -10, 10);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    drawBackground();


    if(!endGame) {

        glDisable(GL_CULL_FACE);
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixd(this->p);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        if(anaglyph){
            double offset;
            offset = 0.01;

            double modelviewTemp[16];

            offsetCamera(offset, modelviewTemp);
            glLoadMatrixd(modelviewTemp);
            filtreRouge();
            drawElements();

            offsetCamera(-offset, modelviewTemp);
            glLoadMatrixd(modelviewTemp);
            filtreBleu();
            drawElements();

        }else{
            glLoadMatrixd(this->m);
            drawElements();
        }

    }

    glutSwapBuffers();

    /// Si la partie est terminée renvoie 1 pour signaler l'arret du programme
    if(endGame){
        glutTimerFunc((unsigned int)1000 / MAX_FPS, loop, 1);
    }else{
        glutTimerFunc((unsigned int)1000 / MAX_FPS, loop, 0);
    }


}

void OpenGL::CallBackReshapeFunc(int w, int h){

    this->width = w;
    this->height= h;

    glViewport(0, 0, this->width, this->height);
    CallBackDisplayFunc();
}

void OpenGL::CallBackKeyboardFunc(unsigned char key, int x, int y) {
    if(key == 'q'){
        exit(0);
    }
}

void OpenGL::CallBackIdleFunc(){
    CallBackDisplayFunc();
}

void OpenGL::loadTexture(GLuint id, Mat img) {

    glBindTexture(GL_TEXTURE_2D, id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glTexImage2D(GL_TEXTURE_2D,     /// Type of texture
                 0,                 /// Pyramid level (for mip-mapping) - 0 is the top level
                 GL_RGB,            /// Internal colour format to convert to
                 img.cols,          /// Image width  i.e. 640 for Kinect in standard mode
                 img.rows,          /// Image height i.e. 480 for Kinect in standard mode
                 0,                 /// Border width in pixels (can either be 1 or 0)
                 0x80E0,            /// Valeur correspondante à GL_BGR
                 GL_UNSIGNED_BYTE,  /// Image data type
                 img.data);         /// The actual image data itself

}

void OpenGL::drawAxes(){

    glColor3f(1, 0, 0);
    glBegin(GL_LINE_STRIP);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.75f, 0.25f, 0.0f);
    glVertex3f(0.75f, -0.25f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.75f, 0.0f, 0.25f);
    glVertex3f(0.75f, 0.0f, -0.25f);
    glVertex3f(1.0f, 0.0f, 0.0f);
    glEnd();
    glColor3f(0, 1, 0);
    glBegin(GL_LINE_STRIP);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.75f, 0.25f);
    glVertex3f(0.0f, 0.75f, -0.25f);
    glVertex3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.25f, 0.75f, 0.0f);
    glVertex3f(-0.25f, 0.75f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);
    glEnd();
    glColor3f(0, 0, 1);
    glBegin(GL_LINE_STRIP);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.25f, 0.0f, 0.75f);
    glVertex3f(-0.25f, 0.0f, 0.75f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.25f, 0.75f);
    glVertex3f(0.0f, -0.25f, 0.75f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    glEnd();

    glColor4ub(255, 255, 0, 255);
    glRasterPos3f(1.1f, 0.0f, 0.0f);

    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'x');
    glRasterPos3f(0.0f, 1.1f, 0.0f);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'y');
    glRasterPos3f(0.0f, 0.0f, 1.1f);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'z');

    glColor3f(1, 1, 1);
}

void OpenGL::drawMazeGround(){
    glBindTexture(GL_TEXTURE_2D, textArray[ID_TEXT_MAZE]);

    /// Plateau de jeu
    glBegin(GL_POLYGON);
    glTexCoord2d(0, 1);glVertex3f(0.0f, 0.0f, 0.0f);
    glTexCoord2d(1, 1);glVertex3f(1.0f, 0.0f, 0.0f);
    glTexCoord2d(1, 0);glVertex3f(1.0f, 1.0f, 0.0f);
    glTexCoord2d(0, 0);glVertex3f(0.0f, 1.0f, 0.0f);
    glEnd();
}

void OpenGL::drawBackground() {
    loadTexture(textArray[ID_TEXT_CAM], this->textCam);
    glBegin(GL_POLYGON);
    glTexCoord2d(0, 1);glVertex3f(0.0, 0.0f, -5.0f);
    glTexCoord2d(0, 0);glVertex3f(0.0f, 1.0f, -5.0f);
    glTexCoord2d(1, 0);glVertex3f(1.0f, 1.0f, -5.0f);
    glTexCoord2d(1, 1);glVertex3f(1.0f, 0.0f, -5.0f);
    glEnd();
}

void OpenGL::drawElements() {

    if(!anaglyph) drawMazeGround();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    drawWalls();
    drawFlag();
    glDisable(GL_TEXTURE_2D);
    applicateMaterial();
    glColor3f(1, 1, 1);
    ball->draw();


    GLfloat sol[3][3] = {{0.0f, 0.0f, 0.0f},
                         {1.0f, 0.0f, 0.0f},
                         {0.0f, 1.0f, 0.0f}};
    GLfloat lightPos[4] = {0.0f, 0.0f, 10.0f, 1.0};
    GLfloat ombre[4][4];

    // On utilise les faces avants seulement
    // ... il faudrait l'utiliser tout le temps ... mais il y a des polygones à l'envers :-( !! )
    // ... il y a un "soucis avec ça" ...

    // La matrice de transformation
    shadowMatrix(sol, lightPos, ombre);

    // Ecriture dans le stencil buffer
    // Pour écrire dans le stencil buffer, on utilise ni
    // le test de profondeur et on ne tient pas compte de la couleur
    glDisable(GL_DEPTH_TEST);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    // Tracé dans le stencil buffer (les points du sol à '1')
    glEnable(GL_STENCIL_TEST);
    glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
    glStencilFunc(GL_ALWAYS, 1, 0xffffffff); // c'est ce '1'
    drawMazeGround();

    // On a a nouveau besoin du tampon de profondeur et de la couleur
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    // On va afficher seulement les valeurs '1' du stencil
    glStencilFunc(GL_EQUAL, 1, 0xffffffff); // c'est ce '1'
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

    // Tracé
    // Pour la transparence de l'ombre
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Pour pouvoir utiliser glColor
    glDisable(GL_LIGHTING);
    // Ombre noire, "transparence" -> 0.5
    glColor4f(0.0f, 0.0f, 0.0f, 0.5f);

    glEnable(GL_CULL_FACE);
    glPushMatrix();
    glMultMatrixf((GLfloat *) ombre);
    ball->draw();
    glPopMatrix();

    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glDisable(GL_CULL_FACE);
    glEnable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glDisable(GL_STENCIL_TEST);


}

void OpenGL::offsetCamera(double offset, double *mat) {
    for(int i = 0; i < 16; i++){
        mat[i] = this->m[i];
    }
    mat[12] += offset;

    if(offset > 0) mat[14] += offset;
}

void OpenGL::filtreRouge() {
    glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE);
}

void OpenGL::filtreBleu() {
    glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_TRUE);
}

void OpenGL::applicateMaterial() {
    GLfloat Lemission[4] = {0.023125f, 0.023125f, 0.023125f, 1.0f};
    GLfloat Ldiffuse[4] = {0.2775f, 0.2775f, 0.2775f, 1.0f};
    GLfloat Lspecular[4] = {0.773911f, 0.773911f, 0.773911f, 1.0f};
    GLfloat Lshininess[1] = {89.6f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Lemission);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, Ldiffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, Lspecular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, Lshininess);
}

void OpenGL::drawFlag(){

    glBindTexture(GL_TEXTURE_2D, textArray[ID_TEXT_FLAG]);

    ///poteau du drapeau
    glMatrixMode(GL_MODELVIEW);
    GLUquadric* params;
    params = gluNewQuadric();

    glPushMatrix();
    glTranslated(this->endPoint->x, this->endPoint->y, -FLAG_PIPE_HEIGHT);
    gluCylinder(params,FLAG_PIPE_RADIUS,FLAG_PIPE_RADIUS,FLAG_PIPE_HEIGHT,20,1); //(..., rayon bas, rayon haut, hauteur,  maillage, "stack")
    glPopMatrix();

    ///drap du peau
    glBegin(GL_POLYGON);
    glTexCoord2d(0, 1);glVertex3d(this->endPoint->x - FLAG_PIPE_RADIUS, this->endPoint->y, -FLAG_PIPE_HEIGHT);
    glTexCoord2d(1, 1);glVertex3d(this->endPoint->x - FLAG_PIPE_RADIUS + FLAG_TOP_SIZE, this->endPoint->y, -FLAG_PIPE_HEIGHT);
    glTexCoord2d(1, 0);glVertex3d(this->endPoint->x - FLAG_PIPE_RADIUS + FLAG_TOP_SIZE, this->endPoint->y, -FLAG_PIPE_HEIGHT - FLAG_TOP_SIZE);
    glTexCoord2d(0, 0);glVertex3d(this->endPoint->x - FLAG_PIPE_RADIUS, this->endPoint->y, -FLAG_PIPE_HEIGHT - FLAG_TOP_SIZE);

    glEnd();

}

void OpenGL::applicateLight() {
    //Light
    glEnable(GL_LIGHTING);
    GLfloat LPosition[4] =  { 1.5f, 1.5f, 0.0f, 1.0};
    GLfloat LAmbient[4] =  { 1.4, 1.4, 1.4, 1.0};
    GLfloat LDiffuse[4] =  {2.0, 2.0, 2.0, 1.0};
    GLfloat LSpecular[4] =  {1.0, 1.0, 1.0, 1.0};
    glLightfv(GL_LIGHT0, GL_POSITION, LPosition);   // position
    glLightfv(GL_LIGHT0, GL_AMBIENT, LAmbient );    // couleur de la forme
    glLightfv(GL_LIGHT0, GL_DIFFUSE, LDiffuse);     // couleur de la lumière
    glLightfv(GL_LIGHT0, GL_SPECULAR, LSpecular);   // couleur du reflet
    glEnable(GL_LIGHT0);
}

void OpenGL::shadowMatrix(GLfloat points_plan[3][3], const GLfloat lightPos[4], GLfloat destMat[4][4]) {
    GLfloat planeCoeff[4];
    GLfloat dot;

    //on calcule un vecteur normal à ce plan
    normal(points_plan,planeCoeff);

    // le dernier coefficient se calcule par substitution
    planeCoeff[3] = - ( (planeCoeff[0]*points_plan[2][0]) + (planeCoeff[1]*points_plan[2][1]) + (planeCoeff[2]*points_plan[2][2]));
    dot = planeCoeff[0] * lightPos[0] + planeCoeff[1] * lightPos[1] + planeCoeff[2] * lightPos[2] + planeCoeff[3] * lightPos[3];

    // maintenant, on projette
    // 1ère colonne
    destMat[0][0] = dot - lightPos[0] * planeCoeff[0];
    destMat[1][0] = 0.0f - lightPos[0] * planeCoeff[1];
    destMat[2][0] = 0.0f - lightPos[0] * planeCoeff[2];
    destMat[3][0] = 0.0f - lightPos[0] * planeCoeff[3];

    // 2ème colonne
    destMat[0][1] = 0.0f - lightPos[1] * planeCoeff[0];
    destMat[1][1] = dot - lightPos[1] * planeCoeff[1];
    destMat[2][1] = 0.0f - lightPos[1] * planeCoeff[2];
    destMat[3][1] = 0.0f - lightPos[1] * planeCoeff[3];

    // 3ème colonne
    destMat[0][2] = 0.0f - lightPos[2] * planeCoeff[0];
    destMat[1][2] = 0.0f - lightPos[2] * planeCoeff[1];
    destMat[2][2] = dot - lightPos[2] * planeCoeff[2];
    destMat[3][2] = 0.0f - lightPos[2] * planeCoeff[3];

    // 4ème colonne
    destMat[0][3] = 0.0f - lightPos[3] * planeCoeff[0];
    destMat[1][3] = 0.0f - lightPos[3] * planeCoeff[1];
    destMat[2][3] = 0.0f - lightPos[3] * planeCoeff[2];
    destMat[3][3] = dot - lightPos[3] * planeCoeff[3];
}

void OpenGL::normal(float v[3][3], float out[3]) {

    float v1[3],v2[3];
    static const int x = 0;
    static const int y = 1;
    static const int z = 2;

    // Calcul de 2 vecteurs à partir des 3 points
    v1[x] = v[0][x] - v[1][x];
    v1[y] = v[0][y] - v[1][y];
    v1[z] = v[0][z] - v[1][z];
    v2[x] = v[1][x] - v[2][x];
    v2[y] = v[1][y] - v[2][y];
    v2[z] = v[1][z] - v[2][z];

    // calcul du produit vectoriel
    out[x] = (v1[y]*v2[z] - v1[z]*v2[y]);
    out[y] = (v1[z]*v2[x] - v1[x]*v2[z]);
    out[z] = (v1[x]*v2[y] - v1[y]*v2[x]);

    // on le réduit à un vecteur unité
    vecteurUnite(out);
}

void OpenGL::vecteurUnite(float vector[3]) {
    float length;

    // Calcul de la norme du vecteur
    length = sqrt((vector[0]*vector[0]) + (vector[1]*vector[1]) + (vector[2]*vector[2]));

    if(length == 0.0f) length = 1.0f;  //évite une violente erreur !!!
    vector[0] /= length;
    vector[1] /= length;
    vector[2] /= length;
}

void OpenGL::drawWalls() {

    glPushMatrix();
    glBindTexture(GL_TEXTURE_2D, textArray[ID_TEXT_WALL]);

    glEnable(GL_DEPTH_TEST);

    /// Pour chacune des lignes
    for(auto &wall : this->walls){
        wall.draw();
    }

    glPopMatrix();
}

const vector<Wall> &OpenGL::getWalls() const {
    return walls;
}

void OpenGL::setWalls(const std::vector<Wall> &walls) {
    this->walls = walls;
}

Point2d *OpenGL::getEndPoint(){
    return this->endPoint;
}

void OpenGL::setEndPoint(cv::Point2d *point){
    this->endPoint = point;
}

void OpenGL::setProjectionMatrix(const double *p) {
    for(int i = 0; i < 16; i++){
        this->p[i] = p[i];
    }
}

void OpenGL::setModelviewMatrix(const double *m) {
    for(int i = 0; i < 16; i++){
        this->m[i] = m[i];
    }
}

void OpenGL::setFps(double fps) {
    OpenGL::fps = fps;
}

void OpenGL::startTimer() {
    this->start = time(nullptr);
}


