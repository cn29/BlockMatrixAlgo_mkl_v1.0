//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2016-2017 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================
/*******************************************************************************
*   This example measures performance of computing the real matrix product
*   C=alpha*A*B+beta*C using a triple nested loop, where A, B, and C are
*   matrices and alpha and beta are double precision scalars.
*
*   In this simple example, practices such as memory management, data alignment,
*   and I/O that are necessary for good programming style and high MKL
*   performance are omitted to improve readability.
********************************************************************************/

#define min(x,y) (((x) < (y)) ? (x) : (y))

#include "blockMatrixAlgs.h"

/* Consider adjusting LOOP_COUNT based on the performance of your computer */
/* to make sure that total run time is at least 1 second */


int main()
{
	double *A = NULL, *Uk, *X, *diagv, *y;
	int  n, k, i, s;
	double sigma, beta;
	double s_initial, s_elapsed;
	char trans = 'n';
	double ratios[10];
	for (i = 0; i < 10; i++) {
		ratios[i] = 0.0005*pow(2,i);
	}
	double rtime1[10] = {0}, rtime2[10] = {0}, rates[10] = {0}, t1 = 0.0, *rtime = &t1;
	s = 20;
	n = 20000;
	sigma = 1.0; beta = 0.0;
	printf("\n"
		" This program compares the performance of two algorithms.\n"
		" The basic idea is: B = (A + sigma * Uk * Uk.T), X_(n+1) = B * X_(n) \n"
		" A - n*n,  Uk - n*k, X - n*1 \n"
		" Iterations s = %d, sigma = %.3f", s, sigma);
	X = (double *)mkl_malloc(sizeof(double)*n, ALIGN);
	for (int i = 0; i < n * 1; i++) 
		X[i] = (double)((i + 1) % 200) / 5.34e3;
	int rm = 10;
	printf("\n In total %d iterations for different size k. \n\n", rm);
	for (int ri = 0; ri < rm; ri++) {
		// dimensions
		k = ratios[ri] * n; //400;
		// initialize
		Uk = (double *)mkl_malloc(sizeof(double)*n*k, ALIGN);
		diagv = (double *)mkl_malloc(sizeof(double)*k, ALIGN);
		for (int i = 0; i < n*k; i++)
			Uk[i] = (double)((i + 1) % 200) / 5.34e3;
		for (i = 0; i < k; i++)
			diagv[i] = (double)((i*i) % 200) / 5.34e3;
		// test run, no recording
		y = blockAlgo1(n, k, s, sigma, Uk, X, rtime);
		mkl_free(y);
		printf("\n n = %d, k = %d, ratio = %.4f ", n, k, ratios[ri]);
		for (i = 0; i < LOOP_COUNT; i++) {
			y = blockAlgo1(n, k, s, sigma, Uk, X, rtime);
			mkl_free(y);
			rtime1[ri] += *rtime;
			y = blockAlgo2(n, k, s, sigma, Uk, X, diagv, rtime);
			mkl_free(y);
			rtime2[ri] += *rtime;
		}
		
		rtime1[ri] /= LOOP_COUNT;
		rtime2[ri] /= LOOP_COUNT;
		rates[ri] = rtime1[ri] / rtime2[ri];
		mkl_free(Uk);
		mkl_free(diagv);
	}

	///////////////////////////////////////////////////////////////////////////////////
	printf("\n\n\n Ratio k/n\t ");
	for (int ri = 0; ri < rm; ri++)
		printf(" %2.4f ", ratios[ri]);
	printf("\n Runtime 1\t ");
	for (int ri = 0; ri < rm; ri++)
		printf(" %2.4f ", rtime1[ri]);
	printf("\n Runtime 2\t ");
	for (int ri = 0; ri < rm; ri++)
		printf(" %2.4f ", rtime2[ri]);
	printf("\n time1/time2\t ");
	for (int ri = 0; ri < rm; ri++)
		printf(" %2.4f ", rates[ri]);

	printf("\n\n Example completed. \n\n");
	cin.get();
	return 0;
}