/*
 * author: caleb7023
 */

#ifndef ACT_FUNC_H
#define ACT_FUNC_H

#include "_math.h"

double sigmoid(double x){
    return 1/(1+exp(-x));
}

double swish(double x){
    return x*sigmoid(x);
}


#endif // ACT_FUNC_H