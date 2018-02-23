#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <iostream>
#include "mkl_vsl.h"
#include <time.h>
#include <assert.h>
#include <cmath>
#include <set>
#include <vector>
using namespace std;
#define LOOP_COUNT 5
#define ALIGN 64


double* blockAlgo1(int n, int k, int s, double sigma, double *Uk, double *X, double *rtime);

double* blockAlgo2(int n, int k, int s, double sigma, double *Uk, double *X, double *diagv, double *rtime);