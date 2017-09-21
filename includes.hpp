#ifndef __PARAMETERS_HPP__
#define __PARAMETERS_HPP__
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <random>

#define PI				3.14159265359

// Environment Parameters
#define VIEW_W			16
#define VIEW_H			9

// Video Parameters
#define WINDOW_TITLE	"Video"
#define FULL_SCREEN		0
#define RENDER_TIME		30
#define WINDOW_H		1280
#define WINDOW_V		720
#define Z_PROX			0.001
#define Z_DIST			1000
#define C_FOVY			75
#define TAM_MOUSE		350
#define FATOR_ZOOM		1.02
#define POINTS			100
#define PLOT_MLP		1

// Comandos de Teclado
#define EXIT 			27
#define MOV_LEFT 		'a'
#define MOV_RIGHT 		'd'
#define MOV_BACK 		's'
#define MOV_FORTH 		'w'
#define MOV_DOWN		'q'
#define MOV_UP 			'e'
#define ZOOM_IN 		'f'
#define ZOOM_OUT 		'v'
#define INC_POINT 		'g'
#define DEC_POINT 		'b'
#define RESET_VIEW_POS	'z'
#define PLOT_AXIS		'x'
#define WALK_SPEED		'c'
#define ORTHO			'o'
#define RANDOMIZE		'r'
#define THRES			't'

// Parametros de movimento da Camera
#define WALK_VEL 		0.01
#define RUN_VEL 		0.04

// Parametros de arquivo
#define BUFFER_SIZE 	256

using namespace std;

#endif
