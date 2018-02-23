#include "blockMatrixAlgs.h"

#define LOOP_COUNT 5
#define ALIGN 64

#define CALL_AND_CHECK_STATUS(function, error_message) do { \
          if(function != SPARSE_STATUS_SUCCESS)             \
          {                                                 \
          printf(error_message); fflush(0);                 \
          status = 1;                                       \
          goto memory_free;                                 \
          }                                                 \
} while(0)

// Input: Uk - n*k, X - k*1, simga - scalar, s - loopmax
// Output: OutX - n*s (X1, ..., Xs-1), rtime - runtime
double* blockAlgo1(int n, int k, int s, double sigma, double *Uk, double *X, double *rtime) {
	double *X_next, *X_k, *outX, beta;
	X_next = (double *)mkl_malloc(sizeof(double)*n, ALIGN);
	X_k = (double *)mkl_malloc(sizeof(double)*k, ALIGN);
	beta = 0.0;
	outX = (double *)mkl_malloc(sizeof(double)*n*s, ALIGN);
	int i = 0;

	if (X_k == NULL || Uk == NULL || X == NULL || X_next == NULL) {
		printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
		mkl_free(X_k);
		mkl_free(X_next);
		return NULL;
	}

	long long lln = n;
	long long llk = k;
	long long llone = 1;
	memcpy(X_next, X, sizeof n * 1 * sizeof(double));
	// start time
	double s_initial = dsecnd();
	for (int ss = 0; ss < s; ss++) {
		// Xk = UkT * X_next
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, llk, llone, lln, 1.0, Uk, llk, X_next, llone, 0.0, X_k, llone);
		// X_next = Uk * X_k
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, 1, k, sigma, Uk, k, X_k, 1, 0.0, X_next, 1);
		// save X_next to X_ss
		memcpy(outX + ss * n, X_next, n * 1 * sizeof(double));
	}
	double s_elapsed = (dsecnd() - s_initial);
	*rtime = s_elapsed;
	mkl_free(X_next);
	mkl_free(X_k);
	// mkl_free(outX);
	return outX;
}


// Input: Uk - n*k, X - k*1, diagv - k*1, simga - scalar, s - loopmax
// Output: OutX - n*s (X1, ..., Xs-1), rtime - runtime
double* blockAlgo2(int n, int k, int s, double sigma, double *Uk, double *X, double *diagv, double *rtime) {
	double beta = 0.0;
	double *outX, *d, *W, *W1, *diagW, *bj, *B, *UkB;
	double *diagvs, *diagvpas;
	//vector<vector<double>> diagvs(s, vector<double>(k, 1)), diagvpas(s, vector<double>(k, 1));
	outX	= (double *)mkl_malloc(sizeof(double)*n*s,	ALIGN);
	d		= (double *)mkl_malloc(sizeof(double)*k,	ALIGN);
	diagvs	= (double *)mkl_malloc(sizeof(double)*k*(s+1),	ALIGN);
	diagvpas = (double *)mkl_malloc(sizeof(double)*k*(s+1),	ALIGN);
	W		= (double *)mkl_malloc(sizeof(double)*k,	ALIGN);
	W1		= (double *)mkl_malloc(sizeof(double)*k,	ALIGN);
	diagW	= (double *)mkl_malloc(sizeof(double)*k*k,	ALIGN);
	bj		= (double *)mkl_malloc(sizeof(double)*k,	ALIGN);
	B		= (double *)mkl_malloc(sizeof(double)*k*s,	ALIGN);
	UkB		= (double *)mkl_malloc(sizeof(double)*n*s,	ALIGN);
	int i = 0, incx = 1;
	long long lln = n;
	long long llk = k;
	long long llone = 1;
	// start time
	double s_initial = dsecnd();
	// d = UkT * X
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, llk, llone, lln, 1.0, Uk, llk, X, llone, 0.0, d, llone);
	// main loop
	for (int ss = 0; ss < s; ss++) {
		// diagvs[(0 ~ k-1) + ss*k] = diagv.^(ss+1) 
		// diagvpas[(0 ~ k-1) + ss*k] = diagvpa.^(ss+1) 
		for (int i = 0; i < k; i++) {
			//diagvs[ss+1][i] = diagvs[ss][i] * diagv[i];
			//diagvpas[ss + 1][i] = diagvpas[ss][i] * (diagv[i] + sigma);
			diagvs[(ss+1)*k+i] = diagvs[ss*k+i] * diagv[i];
			diagvpas[(ss + 1)*k + i] = diagvpas[ss*k + i] * (diagv[i] + sigma);
		}
		for (i = 0; i < (k * 1); i++)
			W[i] = 0.0;
		for (int i = 0; i < ss; i++) {
			for (int kk = 0; kk < k; kk++) {
				W[kk] += diagvs[i*k+kk] * diagvpas[(ss - i + 1)*k+kk];
			}
		}
		// W1 = W*sigma
		cblas_daxpy(k, sigma, W, incx, W1, incx);
		for (i = 0; i < (k * 1); i++)
			W1[i] += diagvs[(ss + 1)+i];
		// construct a diagnal matrix 
		for (i = 0; i < (k * 1); i++)
			for (int j = 0; j < (k * 1); j++)
				if (i == j)
					diagW[i*k + j] = W1[i];
				else
					diagW[i*k + j] = 0.0;
		// bj = diagW * d
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, llk, llone, llk, 1.0, diagW, llk, d, llone, 0.0, bj, llone);
		// bj as a column of matrix B
		memcpy(B + ss * k, bj, k * sizeof(double));
	}
	// UkB = Uk * B;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lln, s, llk, 1.0, Uk, llk, B, s, 0.0, UkB, s);
	// copy to output
	memcpy(outX, UkB, n * s * sizeof(double));

	double s_elapsed = (dsecnd() - s_initial);
	*rtime = s_elapsed;
	mkl_free(d);
	mkl_free(diagvs);
	mkl_free(diagvpas);
	mkl_free(B);
	mkl_free(diagW);
	mkl_free(bj);
	mkl_free(W1);
	mkl_free(W);
	
	mkl_free(UkB);
	return outX;
}
