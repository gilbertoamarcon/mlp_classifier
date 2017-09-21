#include "includes.hpp"
#include "mlp.hpp"

// Parametros da MLP
#define I 				2			// Number of inputs
#define J				4			// Number of HL units
#define K				1			// Number of outputs
#define N				100000		// Number of epochs
#define C				0			// Number of init candidates
#define D				1.0000		// Initial step size
#define	P				1000		// Training set size 
#define	LOAD			0			// Load training set size 
#define	WEIGHTS_PATH	"res/WV"	// Weights file name

Mlp mlp;

double s[I*P];
double d[K*P];

int main(int argc, char **argv){

	if(LOAD)
		mlp.load(WEIGHTS_PATH);	
	else{
		mlp.init(I,J,K,N,C,D,1.0);
		mlp.train(s,d,P);
	}

	return 0;
}