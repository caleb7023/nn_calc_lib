/*
 * author: caleb7023
 */


#ifndef _MATH_H
#define _MATH_H

/**
 * Calculate the exponential function using the Taylor series
 * x: the value to calculate the exponential of (e^x)
 * terms: the number of terms to use in the Taylor series
 * \[\exp(x) = \sum_{n=0}^{\infty} \frac{x^n}{n!}\]
 * \[\exp(x) = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \frac{x^4}{4!} + \cdots\]
 */
double exp(double x, unsigned int terms = 100){
    double result = x+1;
    unsigned int factorial = 2; // to memo the factorial
    double x_pow = x;           // to memo the x^n
    for(int i = 2; i < terms; i++){
        factorial *= i + 1; // calculate the factorial
        x_pow *= x;         // calculate the x^n
        result += x_pow/factorial;
    }
    return result;
}

#endif // _MATH_H