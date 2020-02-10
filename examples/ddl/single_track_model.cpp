#include <cmath>

#include "bicycle_model.h"
#include "tire_models.h"
#include "utilities.h"

/* returns bicycle model derivatives given current state x and input u
   x = [r, uy, ux, psi, E, N]
*/
x_vec_T cont_bicycle_model(const x_vec_T& w, const u_vec_T& u, const owl::veh_params& vehicle){
    // Extract state
    double r    = w[0];
    double uy   = w[1];
    double ux   = w[2];
    double psi  = w[3];
    //double E    = w[4];
    //double N    = w[5];
    // Extract input
    double delta = u[0];
    double fxf   = u[1];
    double fxr   = u[2];

    // steering offset
    //delta += 0.5*M_PI/180;

    // Extract vehicle parameters
    // unpack vehicle parameters
    double a    = vehicle.a_m;
    double b    = vehicle.b_m;
    double L    = a + b;
    double m    = vehicle.m_kg;
    double I    = vehicle.Izz_m2kg;
    double h    = vehicle.hcg_m;
    double Caf  = vehicle.Caf_Nprad;
    double Car  = vehicle.Car_Nprad;
    double Cd0  = vehicle.Cd0_N;
    double Cd1  = vehicle.Cd1_Npmps;
    double mu   = vehicle.mu;

    // Forces
    double fzf = 1/L*(m*b*G - h*(fxf+fxr));
    double fzr = 1/L*(m*a*G + h*(fxf+fxr));
    double af = atan2(uy+a*r, ux) - delta; //0.5 deg
    double ar = atan2(uy-b*r, ux);
    double fyf = fiala(af, Caf, mu, fxf, fzf);
    double fyr = fiala(ar, Car, mu, fxr, fzr);
    double fx_drag = Cd0 + Cd1*ux;

    // Equations of Motion
    x_vec_T xdot;
    xdot[0] = 1/I*(a*fyf*cos(delta) + a*fxf*sin(delta)-b*fyr);
    xdot[1] = 1/m*(fyf*cos(delta) + fxf*sin(delta) + fyr) - r*ux;
    xdot[2] = 1/m*(-fyf*sin(delta) + fxf*cos(delta) + fxr -fx_drag) + r*uy;
    xdot[3] = r;
    xdot[4] = -ux*sin(psi) - uy*cos(psi);
    xdot[5] = ux*cos(psi) - uy*sin(psi);
    return xdot;
}

/* returns x(t+dT) for x(t) = [r, uy, ux, psi, E, N], useful for simulation */
x_vec_T sim_bicycle_model(x_vec_T w, const u_vec_T& u, const double dT, const owl::veh_params& vehicle) {
    bool veh_stopped = w[2] < 0.01;
    if (veh_stopped) {
        w[0] = 0;
        w[1] = 0;
        w[2] = 0;
    }

    // call existing function to set derivates for r, uy, ux, psi
    x_vec_T dwdt = cont_bicycle_model(w, u, vehicle);

    // if stopped, overwrite velocity derivatives to zero
    if (veh_stopped) {
        dwdt[0] = 0;
        dwdt[1] = 0;
        dwdt[2] = 0;
    }
    return w + dT*dwdt;
}

