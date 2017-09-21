#include "includes.hpp"
#include "mlp.hpp"

// Parameters
#define I 				5					// Number of inputs
#define J				10					// Number of HL units
#define K				2					// Number of outputs
#define N				10000				// Number of epochs
#define C				0					// Number of initial candidates
#define D				0.0100				// Initial step size
#define	P				200					// Training set size 
#define	LOAD			0					// Load training set size 
#define	WEIGHTS_PATH	"res/weights"		// Weights file name
#define	TRAIN_SET_FILE	"data/train1.csv"	// Train set file name

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
			while(fileBuffer[aux] != ',' && fileBuffer[aux] != '\n') aux++;
		}
		for(int k = 0; k < K; k++){
			d[p*K + k] = atof(fileBuffer+aux);
			while(fileBuffer[aux] != ',' && fileBuffer[aux] != '\n') aux++;
		}
	}

	fclose(file);

	return 0;

}

int main(int argc, char **argv){

	Mlp mlp;

	double s[I*P];
	double d[K*P];

	load_data(s,d,TRAIN_SET_FILE);

	if(LOAD)
		mlp.load(WEIGHTS_PATH);	
	else{
		mlp.init(I,J,K,N,C,D,1.0);
		mlp.train(s,d,P);
		mlp.store(WEIGHTS_PATH);
	}

	return 0;
}