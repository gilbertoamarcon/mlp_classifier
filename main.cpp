#include "includes.hpp"
#include "mlp.hpp"
#include "GL/freeglut.h"
#include "GL/gl.h"

// Parametros da MLP
#define I 				2		// Número de entradas
#define J				4		// Número de neurônios na HL
#define K				1		// Número de saídas
#define N				100000	// Número de épocas
#define C				96		// Número de candidatos a init
#define D				1.0000	// Tamanho inicial do passo
#define Dx				0.0001	// Taxa de aumento do tamanho do passo
#define	P				1000	// Tamanho do conjunto de treino
#define	LOAD			0		// Carregar conjunto de treino
#define	WEIGHTS_PATH	"WV"	// Nome do arquivo de pesos
#define	STD_DEV			0.10	// Desvio padrao do ruido

Mlp mlp;

double s[I*P];
double d[K*P];

void movMouse(int x, int y);

void atualiza(int n);

void renderizaCena();

void iniGl();

void teclaPressionada(unsigned char key, int x, int y);

void teclaLiberada(unsigned char key, int x, int y);

void plotEixo();

void plotF();

void plotMlpColors();

void plotMlpPoints();

void keyPressed(unsigned char key, int x, int y);

// Variaveis de movimento da Camera
float walk_fast		= 1;
int plot_axis		= 1;
int point_size		= 3;
int zoomOut			= 0;
int zoomIn			= 0;
float point_colors	= 2;
int ortho			= 1;
int thres			= 0;
float movL			= 0;
float movR			= 0;
float movB			= 0;
float movF			= 0;
float movU			= 0;
float movD			= 0;
float theta			= 90;
float phi			= 90;

// Variaveis de posicao da Camera
float cam_vel		= 0;
float camX			= 0;
float camY			= 0;	
float camZ			= 0;
float ctrX			= 0;
float ctrY			= 0;
float ctrZ			= 0;
float cam_rad		= 4;
int camera_topo		= 0;
float c_aspect		= (float)WINDOW_H/WINDOW_V;

int main(int argc, char **argv){

	srand(clock());
	default_random_engine generator;
	normal_distribution<double> dist(0,STD_DEV);

	for(int i = 0; i < P; i++){
			int x1 		= rand()%7-3;
			int x2 		= rand()%7-3;
			s[2*i]		= (double)x1 + dist(generator);
			s[2*i+1]	= (double)x2 + dist(generator);
			// d[i]		= x1 == 1 || x2 == 1;
			d[i]		= x1 == x2 || x1 == -x2;
			// d[i]		= (x1%2 && x2%2);
			// d[i]		= abs(x1) == 1 || abs(x2) == 1;
			// d[i]		= x1 == x2 || x1 == -x2 || abs(x1) == 1 || abs(x2) == 1;
			// d[i]		= (x1 == 1 && x2 == 2) ||(x1 == 1 && x2 == -2) ||(x1 == 3 && x2 == 2) ||(x1 == -2 && x2 == 0) ||(x1 == 1 && x2 == 1);
	}

	if(LOAD)
		mlp.load(WEIGHTS_PATH);	
	else{
		mlp.init(I,J,K,N,C,D,Dx,1.0);
		mlp.train(s,d,P);
	}

	glutInit(&argc, argv);

	iniGl();

	glutMainLoop();

	return 0;
}

void plotF(){
	glColor3f(1.0,1.0,1.0);
	glPointSize(point_size);
	glBegin(GL_POINTS);
		for (int p = 0; p < P; p++)
			if(ortho){
				glColor3f(d[p],0.0,1-d[p]);
				glVertex3f(s[2*p],s[2*p+1],-2);
			}
			else
				glVertex3f(s[2*p],s[2*p+1],d[p]);
	glEnd();
	glPointSize(1);
}

void plotMlpPoints(){
	glColor3f(0.0,1.0,1.0);
	glBegin(GL_POINTS);
		for (int x = 0; x < POINTS; x++){
			for (int y = 0; y < POINTS; y++){
				mlp.x[0] = VIEW_H*((float)x/POINTS-0.50);
				mlp.x[1] = VIEW_H*((float)y/POINTS-0.50);
				mlp.eval();
				double mlp_v = mlp.o[0];
				if(thres) mlp_v = (mlp_v >= 0.5)?1:0;
				glVertex3f(mlp.x[0],mlp.x[1],mlp_v);
			}
		}
	glEnd();
}

void plotMlpColors(){

	double m[POINTS][POINTS];
	for(int x = 0; x < POINTS; x++){
		for(int y = 0; y < POINTS; y++){
			mlp.x[0] = VIEW_W*((float)x/POINTS-0.50);
			mlp.x[1] = VIEW_H*((float)y/POINTS-0.50);
			mlp.eval();
			m[x][y] = mlp.o[0];
		}
	}

	glBegin(GL_QUADS);
		for (int x = 0; x < POINTS-1; x++){
			for (int y = 0; y < POINTS-1; y++){
				double x0 = VIEW_W*((float)(x+0)/POINTS-0.50);
				double x1 = VIEW_W*((float)(x+1)/POINTS-0.50);
				double y0 = VIEW_H*((float)(y+0)/POINTS-0.50);
				double y1 = VIEW_H*((float)(y+1)/POINTS-0.50);
				double mlp_v = 0;

				mlp_v = m[x+0][y+0];
				if(thres) mlp_v = (mlp_v >= 0.5)?1:0;
				glColor3f(0.0+mlp_v,1.0,1.0-mlp_v);
				glVertex3f(x0,y0,mlp_v);

				mlp_v = m[x+1][y+0];
				if(thres) mlp_v = (mlp_v >= 0.5)?1:0;
				glColor3f(0.0+mlp_v,1.0,1.0-mlp_v);
				glVertex3f(x1,y0,mlp_v);

				mlp_v = m[x+1][y+1];
				if(thres) mlp_v = (mlp_v >= 0.5)?1:0;
				glColor3f(0.0+mlp_v,1.0,1.0-mlp_v);
				glVertex3f(x1,y1,mlp_v);

				mlp_v = m[x+0][y+1];
				if(thres) mlp_v = (mlp_v >= 0.5)?1:0;
				glColor3f(0.0+mlp_v,1.0,1.0-mlp_v);
				glVertex3f(x0,y1,mlp_v);
			}
		}
	glEnd();
}

void renderizaCena(){

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if(ortho)
		glOrtho(-VIEW_W/2,VIEW_W/2,-VIEW_H/2,VIEW_H/2,3,-3);
	else
		gluPerspective(C_FOVY,c_aspect,Z_PROX,Z_DIST);
	glMatrixMode(GL_MODELVIEW);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushMatrix();
		if(ortho){
			if(PLOT_MLP)
				plotMlpColors();			
		}else{
			gluLookAt(ctrX+camX,ctrY+camY,ctrZ+camZ,ctrX,ctrY,ctrZ,0,0,camera_topo);
			if(PLOT_MLP)
				plotMlpPoints();			
		}
		if(plot_axis) plotEixo();
		plotF();
	glPopMatrix();

	glutSwapBuffers();
}

void iniGl(){
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_H, WINDOW_V);
	glutCreateWindow(WINDOW_TITLE);
	glutKeyboardFunc(&teclaPressionada);
	glutKeyboardUpFunc(&teclaLiberada);
	glutDisplayFunc(&renderizaCena);
	glutIdleFunc(&renderizaCena);
	glutPassiveMotionFunc(&movMouse);
	glClearColor(0,0,0,0);
	glDisable(GL_CULL_FACE);
	glDisable(GL_ALPHA_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glutTimerFunc(1,atualiza,0);
	glutSetCursor(GLUT_CURSOR_NONE);
	glutFullScreen();
}

void movMouse(int x, int y){

	int relMouseX = x - WINDOW_H/2;
	int relMouseY = y - WINDOW_V/2;

	if(relMouseX > TAM_MOUSE)
		glutWarpPointer(WINDOW_H/2 - TAM_MOUSE, y);
	else if(relMouseX < - TAM_MOUSE)
		glutWarpPointer(WINDOW_H/2 + TAM_MOUSE, y);

	if(relMouseY > TAM_MOUSE)
		glutWarpPointer(x, WINDOW_V/2 - TAM_MOUSE);
	else if(relMouseY < - TAM_MOUSE)
		glutWarpPointer(x, WINDOW_V/2 + TAM_MOUSE);

	theta	= 180*(double)relMouseX/TAM_MOUSE + 180;
	phi		= 89*(double)relMouseY/TAM_MOUSE + 270;

}

void plotEixo(){
	glLineWidth(point_size);
	glColor3f(1,0.5,0.5);
	glBegin(GL_LINES);
		glVertex3f(0,0,0);
		glVertex3f(1,0,0);
	glEnd();
	glColor3f(0.5,1,0.5);
	glBegin(GL_LINES);
		glVertex3f(0,0,0);
		glVertex3f(0,1,0);
	glEnd();
	glColor3f(0.5,0.5,1);
	glBegin(GL_LINES);
		glVertex3f(0,0,0);
		glVertex3f(0,0,1);
	glEnd();
	glLineWidth(1	);
}

void atualiza(int n){
	glutTimerFunc(1,atualiza,0);
	if(zoomIn)
		cam_rad /= FATOR_ZOOM;
	if(zoomOut)
		cam_rad *= FATOR_ZOOM;
	camX	= cam_rad*sin(PI*theta/180.0)*sin(PI*phi/180.0);
	camY	= cam_rad*cos(PI*theta/180.0)*sin(PI*phi/180.0);
	camZ	= cam_rad*cos(PI*phi/180.0);
	camera_topo	= 2*(phi>180)-1;
	if(movL){
		ctrX -= cam_vel*cos(PI*theta/180.0);
		ctrY += cam_vel*sin(PI*theta/180.0);
	}
	if(movR){
		ctrX += cam_vel*cos(PI*theta/180.0);
		ctrY -= cam_vel*sin(PI*theta/180.0);
	}
	if(movB){
		ctrX -= cam_vel*sin(PI*theta/180.0);
		ctrY -= cam_vel*cos(PI*theta/180.0);
	}
	if(movF){
		ctrX += cam_vel*sin(PI*theta/180.0);
		ctrY += cam_vel*cos(PI*theta/180.0);
	}
	if(movD)
		ctrZ -= cam_vel;
	if(movU)
		ctrZ += cam_vel;
	if(walk_fast)
		cam_vel = RUN_VEL;
	else
		cam_vel = WALK_VEL;
}

void teclaPressionada(unsigned char key, int x, int y){
	switch (key){
		case EXIT:
			mlp.store(WEIGHTS_PATH);
			exit(0);
			break;
		case WALK_SPEED:
			walk_fast = !walk_fast;
			break;
		case PLOT_AXIS:
			plot_axis = !plot_axis;
			break;
		case ORTHO:
			ortho = !ortho;
			break;
		case THRES:
			thres = !thres;
			break;
		case RANDOMIZE:			
			mlp.randomize();
			break;
		case RESET_VIEW_POS:
			ctrX = 0;
			ctrY = 0;
			ctrZ = 0;
			break;
		case INC_POINT:
			point_size++;
			break;
		case DEC_POINT:
			point_size--;
			if(point_size < 1)
				point_size = 1;
			break;
		case ZOOM_IN:
			zoomIn = 1;
			break;
		case ZOOM_OUT:
			zoomOut = 1;
			break;
		case MOV_LEFT:
			movL = 1;
			break;
		case MOV_RIGHT:
			movR = 1;
			break;
		case MOV_BACK:
			movB = 1;
			break;
		case MOV_FORTH:
			movF = 1;
			break;
		case MOV_DOWN:
			movU = 1;
			break;
		case MOV_UP:
			movD = 1;
			break;
		default:
		break;
	}
}

void teclaLiberada(unsigned char key, int x, int y){
	switch (key){
		case ZOOM_IN:
			zoomIn = 0;
			break;
		case ZOOM_OUT:
			zoomOut = 0;
			break;
		case MOV_LEFT:
			movL = 0;
			break;
		case MOV_RIGHT:
			movR = 0;
			break;
		case MOV_BACK:
			movB = 0;
			break;
		case MOV_FORTH:
			movF = 0;
			break;
		case MOV_DOWN:
			movU = 0;
			break;
		case MOV_UP:
			movD = 0;
			break;
		default:
			break;
	}
}
