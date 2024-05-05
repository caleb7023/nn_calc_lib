/*
 * author: caleb7023
 */

#ifndef _MATH_H
#define _MATH_H

double exp(double x){
    double result = 1;
    for(int i = 0; i < 100; i++){
        double term = 1;
        for(int j = 1; j <= i; j++){
            term *= x / j;
        }
        result += term;
    }
    return result;
};

#endif // _MATH_H