#include "includes.hpp"
#include "mlp.hpp"

// Parameters
#define I 				5					// Number of inputs
#define J				10					// Number of HL units
#define K				2					// Number of outputs
#define N				1000000				// Number of epochs
#define C				0					// Number of initial candidates
#define D				0.0050				// Initial step size
#define	P				200					// Training set size 
#define	LOAD			1					// Load training set size 
#define	WEIGHTS_PATH	"res/weights"		// Weights file name
#define	TRAIN_SET_FILE	"data/train1.csv"	// Train set file name
#define	TEST_SET_FILE	"data/test2.csv"	// Test set file name

int load_data(double *s, double *d, char *filename){

	FILE *file;	
	char fileBuffer[BUFFER_SIZE]; 

	file 	= fopen(filename,"r");
	if(file == NULL)
		return 1;

	for(int p = 0; p < P; p++){		
		if(fgets(fileBuffer, BUFFER_SIZE, file) == NULL){
			printf("Error: Wrong file type.\n");
			return 1;
		}
		int aux = 0;
		for(int i = 0; i < I; i++){
			s[p*I + i] = atof(fileBuffer+aux);
			while(fileBuffer[aux++] != ',');
		}
		for(int k = 0; k < K; k++){
			d[p*K + k] = atof(fileBuffer+aux);
			while(fileBuffer[aux++] != ',');
		}
	}

	fclose(file);

	return 0;

}

double test(Mlp mlp, double *s, double *d){

	int score = 0;
	for(int p = 0; p < P; p++){

		// Loading inputs
		for(int i = 0; i < I; i++)
			mlp.x[i] = s[p*I+i];

		// Feedforward
		mlp.eval();

		// Computing score
		for(int k = 0; k < K; k++)
			score += (abs(d[p*K + k]-mlp.o[k]) < 0.5);

	}

	return 100*score/(P*K);

}

int main(int argc, char **argv){

	Mlp mlp;

	double s[I*P];
	double d[K*P];


	if(LOAD){
		load_data(s,d,TEST_SET_FILE);
		mlp.load(WEIGHTS_PATH);
		printf("%5.2f %\n", test(mlp,s,d));
	}else{
		load_data(s,d,TRAIN_SET_FILE);
		mlp.init(I,J,K,N,C,D,1.0);
		mlp.train(s,d,P);
		printf("%5.2f %\n", test(mlp,s,d));
		mlp.store(WEIGHTS_PATH);
	}

	return 0;
}