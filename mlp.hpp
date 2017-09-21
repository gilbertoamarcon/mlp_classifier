#ifndef __MLP_HPP__
#define __MLP_HPP__
#include "includes.hpp"

struct Weights{
		
		double E;

		// Pesos
		double *V;
		double *W;

		Weights(int I,int J,int K){
			E = 1e99;
			V = new double[J*(I+1)];
			W = new double[K*(J+1)];
		}

		~Weights(){
			free(V);
			free(W);
		}
};

class Mlp{

	private:
		int I;		// Número de entradas
		int J;		// Número de neurônios na HL
		int K;		// Número de saídas
		int N;		// Número de épocas
		int C;		// Número de candidatos a ponto de inicio
		double D;	// Tamanho inicial do passo
		double Dx;	// Taxa de crescimento do tam do passo
		double iniRange;

		// Pesos
		Weights *mainW;	// Pesos principais
		Weights *bkpW1;	// Pesos backup 1
		Weights *bkpW2;	// Pesos backup 2

		// Variáveis Internas
		double *u;
		double *y;
		double *delta_o;
		double *delta_h;

		default_random_engine generator;

		// Avalia erro dos pesos atuais
		void evalError(double *s, double *d, int P);

		// Iteracao do algoritmo gradiente
		void itTrain(double *s, double *d, int P,double stepSize);

		// Realiza copia de backup dos pesos
		void backupWeights(Weights *bkpWeights);

		// Restaura copia de backup dos pesos
		void restoreBackupWeights(Weights *bkpWeights);

		// Descida gradiente
		void descend(double *s, double *d, int P,int N);

	public:

		// Entradas
		double *x;

		// Saídas
		double *o;

		// Construtor
		Mlp();

		void init(int I,int J,int K,int N,int C,double D,double Dx,double iniRange);

		// Inicialização dos pesos
		void randomize();

		// Avaliação da saída para dada entrada
		void eval();

		// Processo de treinamento
		void train(double *s,double *d, int P);

		// Armazenamento dos pesos
		int store(char *mlp_weights);

		// Carregamento dos pesos
		int load(char *mlp_weights);
};


#endif
