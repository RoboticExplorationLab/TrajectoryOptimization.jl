#pragma once

#include <owl/veh_params.h>

#include "mpc_types.h"
#include "world.h"
#include "utilities.h"

/* returns bicycle model derivatives given current state w and input u */
x_vec_T cont_bicycle_model(const x_vec_T& w, const u_vec_T& u,
                           const owl::veh_params& vehicle, double);

/* returns x(t+dT) for x(t) = [r, uy, ux, psi, E, N], useful for simulation */
x_vec_T sim_bicycle_model(x_vec_T w, const u_vec_T& u, const double dT,
                          const owl::veh_params& vehicle, double);

