#include <cmath>

#include "tire_models.h"
#include "mpc_types.h"
#include "utilities.h"

/* returns fy according to the fiala tire model */
double fiala(double alpha_rad, double Ca_Nprad, double mu, double fx_N, double fz_N) {
    double tanAlpha = std::tan(alpha_rad);
    double fy_max_N = calc_fy_max(mu, fx_N, fz_N);
    double fy;
    double t1, t2, t3;
    if (std::abs(alpha_rad) < std::atan2(3*fy_max_N, Ca_Nprad)) {
        t1 = - Ca_Nprad                                         * tanAlpha;
        t2 =  std::pow(Ca_Nprad,2)/(3*fy_max_N)                 * tanAlpha*std::abs(tanAlpha);
        t3 = - std::pow(Ca_Nprad,3)/(9*std::pow(fy_max_N,2))    * 1/3*std::pow(tanAlpha,3);
        fy = t1 + t2 + t3;
    } else {
        //std::cout << "TIRES SATURATED" << std::endl;
        fy = -fy_max_N * sign(alpha_rad);
    }
    //fy = -Ca_Nprad*alpha_rad;
    //std::cout << "fiala: alpha=" << alpha_rad << ", Ca=" << Ca_Nprad << ", mu=" << mu << ", fx=" << fx_N << ", fz=" << fz_N << std::endl;
    //std::cout << "\tfy=" << fy << ", maxFy=" << fy_max_N << ", tanA=" << tanAlpha << ", t1=" << t1 << ", t2=" << t2
        //<< ", t3=" << t3 << std::endl << std::endl;
    return fy;
}

/* returns max fy, derating by fx */
double calc_fy_max(double mu, double fx_N, double fz_N) {
    double fy_max_N;
    // calc fx_max by derating based on fx
    if (std::pow(mu*fz_N,2) > std::pow(fx_N,2)) {
        fy_max_N = std::sqrt( std::pow(mu*fz_N,2) - std::pow(fx_N,2) );
    } else {
        // even when the tire is under full acceleration or braking, we assume we can still get SAT_TIRE_FY_N of lateral force
        fy_max_N = SAT_TIRE_FY_N;
    }
    return fy_max_N;
}

