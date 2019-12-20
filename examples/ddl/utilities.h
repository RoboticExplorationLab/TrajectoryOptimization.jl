#pragma once

#include <cmath>
#include <vector>

//#include "mpc_types.h"

/* Constants */
constexpr double G = 9.81;  // m/s^2
constexpr double DEG_TO_RAD = M_PI/180;
constexpr double RAD_TO_DEG = 180/M_PI;


/* computes x(i+1) - x(i) for each i in x, returns array of length one less than length of x */
template<typename T>
void diff(const T* x, T* y, const size_t length){
    for (size_t i=0; i<(length-1); i++) {
        y[i] = x[i+1] - x[i];
    }
}

/* returns sign of x, 0 if x=0 */
template<typename T>
T sign(const T& x) {
    if (x > 0) {
        return 1;
    } else if (x < 0) {
        return -1;
    } else {
        return 0;
    }
}

/* Performs a binary search and returns the index of the element closest but not exeeding xq in vector x. Assumes xq is
 * within bounds of the vector and that x is monotonically increasing.
 * Templated to be called with any Container type that supports operator[], size is the number of elements in container.
 */
template<typename Container>
size_t binarySearch(const Container& x, size_t size, double xq) {
    size_t low_ind = 0;
    size_t high_ind = size-1;
    size_t mid_ind;

    while (true) {
        mid_ind = floor((low_ind+high_ind)/2);

        if(low_ind > high_ind){
            //std::cerr << "Search Error" << std::endl;
            return -1;
        }
        // check if we are done
        if ( (high_ind-low_ind) == 1 ) {
            break;
        }
        if (x[mid_ind] < xq) {
            low_ind = mid_ind;
        } else {
            high_ind = mid_ind;
        }
    }
    return low_ind;
}

/* interpolation helper struct for constant time interpolation */
struct InterpData_T {
    size_t low_idx;
    double frac;
    InterpData_T(size_t low, double frac_): low_idx(low), frac(frac_) {}
    //InterpData_T(): low_idx(0), frac(0) {}
};

/* 1D interpolation
 * x is of type Container, which supports operator[]
 * y is the array of values which correspond to those in x
 * size is the number of elements in each of x and y. x[size-1] is the last element in x
 * xq is the x query value
 * yq is the y query value
 * InterpData_T is returned, allowing constant time interpolation calls to replace:
 *      "interpolate1D(x, other_y, size, xq, other_yq)"
 * with:
 *      "const_time_interp(InterpData, other_y, other_yq)"
 */
template<typename Container1, typename Container2>
InterpData_T interpolate1D(const Container1& x, const Container2& y, size_t size, const double xq, double& yq) {
    // first check if query point exceeds bounds of vector, saturate if so
    if (xq <= x[0]) {
        yq = y[0];
        return InterpData_T(0, 0);
    }else if (xq >= x[size-1]) {
        yq = y[size-1];
        return InterpData_T(size-2, 1);
    }

    // find lower index
    size_t low_ind = binarySearch(x, size,  xq);
    double frac = (xq-x[low_ind])/(x[low_ind+1]-x[low_ind]);
    InterpData_T interp(low_ind, frac);

    const_time_interp(interp, y, yq);
    return interp;
}

template<typename Container>
void const_time_interp(const InterpData_T& interp, const Container& y, double& yq) {
    yq = y[interp.low_idx] + interp.frac*(y[interp.low_idx+1]-y[interp.low_idx]);
}

template<typename T>
T simple_interp(const double frac, const T low, const T high) {
    return (1-frac)*low + frac*high;
}

template<typename T>
T wrap_to_pi(const T x) {
    double y = x;
    while (y > M_PI) {
            y -= 2*M_PI;
    }
    while (y < -M_PI) {
            y += 2*M_PI;
    }
    return y;
}
