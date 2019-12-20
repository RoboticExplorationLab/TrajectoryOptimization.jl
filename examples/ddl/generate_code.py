import numpy as np
from casadi import *

# *** Author: Matt Brown, adapted by John Alsterda ***
################################################################################
# Constants
c                = {}    # constant not parameters
c["N"]           = 30    # number of stages in the optimization
c["N_S"]         = 16    # number of states: ( delta fx r uy ux psi e t )*2
c["N_U"]         =  4    # number of inputs: ( delta_dot fx_dot )*2
c["N_REPLAN"]    =  1    # number of stages to use measured nom mu

c['N_PDYN']      = 12    # number of parameters passed to spatial_dyn
c['N_PCOST']     = 23    # number of parameters passed to stagewise_cost

# scaling factors
c["scale_fx"]    = 1000
c["scale_ux"]    = 10

c["ODE_mode"]    = 'dyn' # 'kin'
c["max_fx_frac"] = 0.8

################################################################################
#Define Functions

def py_fiala_tire_model(a, Ca, mu, fz, fx):
    # enforce fx constraints
    #fx       = fmin( cos(a)*mu*fz, fx)
    #fx       = fmax(-cos(a)*mu*fz, fx)
    fx       = fmin( 0.99*mu*fz, fx) # 99% to prevent Fiala divide by zero
    fx       = fmax(-0.99*mu*fz, fx) #   unnecessary? or move 99% to sp_dyn()?

    fy_max   = sqrt( (mu*fz)**2 - fx**2 )
    a_slide  = atan2(3*fy_max, Ca)
    tan_a    = tan(a)

    fy_unsat = ( -Ca * tan_a + Ca**2/(3*fy_max) * tan_a * fabs(tan_a)
                             - Ca**3/(27*fy_max**2) * tan_a**3 )
    fy_sat   = -fy_max * sign(a)

    fy       = if_else(fabs(a) < a_slide, fy_unsat, fy_sat)
    return fy
# construct casadi function, SX
temp_alpha  = SX.sym('temp_alpha', 1)
temp_Ca     = SX.sym('temp_Ca', 1)
temp_mu     = SX.sym('temp_mu', 1)
temp_fz     = SX.sym('temp_fz', 1)
temp_fx     = SX.sym('temp_fx', 1)
fiala_tire_model = Function('fiala_tire_model', [temp_alpha, temp_Ca, temp_mu, temp_fz, temp_fx],
        [py_fiala_tire_model(temp_alpha, temp_Ca, temp_mu, temp_fz, temp_fx)],
        ['alpha', 'Ca', 'mu', 'fz', 'fx'], ['fy'])

def py_logit_tire_model(a, Ca, mu, fz, fx):
    fy_max = sqrt( (mu*fz)**2 - fx**2 )
    gain = 2*Ca/(fy_max)
    fy = fy_max*(1 - 2*exp(gain*a)/(1 + exp(gain*a)))
    return fy
temp_alpha  = SX.sym('temp_alpha', 1)
temp_Ca     = SX.sym('temp_Ca', 1)
temp_mu     = SX.sym('temp_mu', 1)
temp_fz     = SX.sym('temp_fz', 1)
temp_fx     = SX.sym('temp_fx', 1)
logit_tire_model = Function('logit_tire_model', [temp_alpha, temp_Ca, temp_mu, temp_fz, temp_fx],
        [py_logit_tire_model(temp_alpha, temp_Ca, temp_mu, temp_fz, temp_fx)],
        ['alpha', 'Ca', 'mu', 'fz', 'fx'], ['fy'])

# converts to physical units (N, mps, etc)
def py_scale(z):
    del_dot      = z[ 0]                 # u
    fx_dot       = z[ 1] * c["scale_fx"]
    del_dot_c    = z[ 2]                 # u_c
    fx_dot_c     = z[ 3] * c["scale_fx"]
    delta        = z[ 4]                 # x
    fx           = z[ 5] * c["scale_fx"]
    r            = z[ 6]
    uy           = z[ 7]
    ux           = z[ 8] * c["scale_ux"]
    dpsi         = z[ 9]
    e            = z[10]
    t            = z[11]
    delta_c      = z[12]                 # x_c
    fx_c         = z[13] * c["scale_fx"]
    r_c          = z[14]
    uy_c         = z[15]
    ux_c         = z[16] * c["scale_ux"]
    dpsi_c       = z[17]
    e_c          = z[18]
    t_c          = z[19]
    return vertcat( del_dot, fx_dot, del_dot_c, fx_dot_c,
                    delta,   fx,   r,   uy,   ux,   dpsi,   e,   t,
                    delta_c, fx_c, r_c, uy_c, ux_c, dpsi_c, e_c, t_c )
temp_z = SX.sym('temp_z', c['N_S']+c['N_U'])
scale = Function('scale', [temp_z], [py_scale(temp_z)], ['z'], ['z_si'])

# assumes fx in kN, ux is scaled by scale_ux
# Important to have well-scaled problem esp. if using approx Hessian (BFGS). 
# NOTE that since the sys becomes more stiff at lower speeds, taking smaller 
# integration steps is valuable in those regimes
def py_spatial_dynamics(y, u, p_dyn):
    # y = [ delta, fx, r, uy, ux, dpsi, e, t ]
    # p = [ s_m,    k_r,    e_des,    ux_des, e_min,  e_max,
    #       obs1_s, obs1_e, obs1_rad, obs2_s, obs2_e, obs2_rad ]
    # returns y_prime
    #   one fx, gets split between axles

    z = scale( vertcat(u, y) )
    delta_dot   = z[ 0]                 # u
    fx_dot      = z[ 1]
    delta_dot_c = z[ 2]                 # u_c
    fx_dot_c    = z[ 3]
    delta       = z[ 4]                 # x
    fx          = z[ 5]
    r           = z[ 6]
    uy          = z[ 7]
    ux          = z[ 8]
    dpsi        = z[ 9]
    e           = z[10]
    t           = z[11]
    delta_c     = z[12]                 # x_c
    fx_c        = z[13]
    r_c         = z[14]
    uy_c        = z[15]
    ux_c        = z[16]
    dpsi_c      = z[17]
    e_c         = z[18]
    t_c         = z[19]

    k           = p_dyn[0]
    a           = p_dyn[1]
    b           = p_dyn[2]
    h           = p_dyn[3]
    m           = p_dyn[4]
    Iz          = p_dyn[5]
    Cf          = p_dyn[6]
    Cr          = p_dyn[7]
    mu          = p_dyn[8]
    mu_c        = p_dyn[9]
    Cd0_N       = p_dyn[10]
    Cd1_Npmps   = p_dyn[11]
    l           = a + b
    g           = 9.81

    # s_dot factor to convert between time and space
    s_dot     = (ux  *cos(dpsi  ) - uy  *sin(dpsi  )) / (1-k*e  )
    s_dot_c   = (ux_c*cos(dpsi_c) - uy_c*sin(dpsi_c)) / (1-k*e_c)
    t_prime   = 1/s_dot
    t_prime_c = 1/s_dot_c

    delta_prime   = 1/s_dot   * delta_dot
    fx_prime      = 1/s_dot   * fx_dot
    delta_prime_c = 1/s_dot_c * delta_dot_c
    fx_prime_c    = 1/s_dot_c * fx_dot_c

    # dynamic model
    if c["ODE_mode"] == 'dyn':
        fxf   = fmax( fx  *b/l, fx   )  # FWD GTI fxf in [-f_max*b/l, f_max' ]
        fxr   = fmin( fx  *a/l, 0    )  #         fxr in [-f_max*a/l, 0      ]
        fxf_c = fmax( fx_c*b/l, fx_c )
        fxr_c = fmin( fx_c*a/l, 0    )
        #fxf   = fx*b/l                  # AWD
        #fxr   = fx*a/l
        #fxf_c = fx_c*b/l
        #fxr_c = fx_c*a/l

        #fzf = ( m*b*g - h*2*fx_N ) / l
        #fzr = ( m*a*g + h*2*fx_N ) / l
        fzf  =   m*b*g/l           # no weight transfer
        fzr  =   m*a*g/l

        # nonlinear tire model
        af    = atan2( uy  +a*r  , ux   ) - delta
        ar    = atan2( uy  -b*r  , ux   )
        af_c  = atan2( uy_c+a*r_c, ux_c ) - delta_c
        ar_c  = atan2( uy_c-b*r_c, ux_c )
        #fyf   = fiala_tire_model( af  , Cf, mu  , fzf, fxf   )
        #fyr   = fiala_tire_model( ar  , Cr, mu  , fzr, fxr   )
        #fyf_c = fiala_tire_model( af_c, Cf, mu_c, fzf, fxf_c )
        #fyr_c = fiala_tire_model( ar_c, Cr, mu_c, fzr, fxr_c )
        fyf   = logit_tire_model( af  , Cf, mu  , fzf, fxf   )
        fyr   = logit_tire_model( ar  , Cr, mu  , fzr, fxr   )
        fyf_c = logit_tire_model( af_c, Cf, mu_c, fzf, fxf_c )
        fyr_c = logit_tire_model( ar_c, Cr, mu_c, fzr, fxr_c )

        #fx_drag   = Cd0_N + Cd1_Npmps*ux
        #fx_drag_c = Cd0_N + Cd1_Npmps*ux_c
        fx_drag = Cd0_N # constant drag force

        r_prime      = 1/s_dot   * ( a*(fyf  *cos(delta  ) + fxf  *sin(delta  )) - b*fyr   ) / Iz
        r_prime_c    = 1/s_dot_c * ( a*(fyf_c*cos(delta_c) + fxf_c*sin(delta_c)) - b*fyr_c ) / Iz
        uy_prime     = 1/s_dot   * (-r  *ux   + ( fyf  *cos(delta  ) + fxf  *sin(delta  ) + fyr   ) / m )
        uy_prime_c   = 1/s_dot_c * (-r_c*ux_c + ( fyf_c*cos(delta_c) + fxf_c*sin(delta_c) + fyr_c ) / m )
        ux_prime     = 1/s_dot   * ( r  *uy   + (-fyf  *sin(delta  ) + fxf  *cos(delta  ) + fxr   - fx_drag ) / m )
        ux_prime_c   = 1/s_dot_c * ( r_c*uy_c + (-fyf_c*sin(delta_c) + fxf_c*cos(delta_c) + fxr_c - fx_drag ) / m )
        dpsi_prime   = 1/s_dot   * r   - k
        dpsi_prime_c = 1/s_dot_c * r_c - k
        e_prime      = 1/s_dot   * ( ux  *sin(dpsi  ) + uy  *cos(dpsi  ) )
        e_prime_c    = 1/s_dot_c * ( ux_c*sin(dpsi_c) + uy_c*cos(dpsi_c) )

    # simplified kinematic model #mjb
    """
    r_dot_kin    = 0
    uy_dot_kin   = 0
    ux_dot_kin   = 1/m * 2 * fx_N
    s_dot_kin    = (ux*cos(dpsi) - uy*sin(dpsi)) / (1-k_r*e)
    e_dot_kin    =  ux*sin(dpsi) + uy*cos(dpsi)
    dpsi_dot_kin =  ux/l*tan(delta) - k_r*s_dot
    y_dot_kin    = vertcat( del_dot,                  fx_dot/c["scale_fx"],
                            r_dot_kin,                uy_dot_kin,
                            ux_dot_kin/c["scale_ux"], dpsi_dot_kin,
                            s_dot_kin/c["scale_s"],   e_dot_kin            )
    if c["ODE_mode"] == 'dyn':
        y_dot = if_else(ux < c["UX_THRES"], y_dot_kin, y_dot)
    else:
        y_dot = y_dot_kin
    """
    # kinematic spatial model #jpa
    if c["ODE_mode"] == 'kin':
        r_prime      = 0
        r_prime_c    = 0
        uy_prime     = 0
        uy_prime_c   = 0
        ux_prime     = 1/s_dot   * 1/m * fx
        ux_prime_c   = 1/s_dot_c * 1/m * fx_c
        dpsi_prime   = 1/s_dot   * ux  /l*tan(delta  ) - k
        dpsi_prime_c = 1/s_dot_c * ux_c/l*tan(delta_c) - k
        e_prime      = 1/s_dot   * ( ux  *sin(dpsi  ) + uy  *cos(dpsi  ) )
        e_prime_c    = 1/s_dot_c * ( ux_c*sin(dpsi_c) + uy_c*cos(dpsi_c) )

    y_prime = vertcat( delta_prime,   fx_prime/c["scale_fx"],   r_prime,
                       uy_prime,      ux_prime/c["scale_ux"],   dpsi_prime,
                       e_prime,       t_prime,
                       delta_prime_c, fx_prime_c/c["scale_fx"], r_prime_c,
                       uy_prime_c,    ux_prime_c/c["scale_ux"], dpsi_prime_c,
                       e_prime_c,     t_prime_c )
    return y_prime
# construct casadi function, SX for continuous dynamics
temp_y = SX.sym('temp_y', c['N_S'])
temp_u = SX.sym('temp_u', c['N_U'])
temp_p = SX.sym('temp_p', c['N_PDYN'])
spatial_dynamics = Function( 'spatial_dynamics', [temp_y, temp_u, temp_p],
                             [py_spatial_dynamics(temp_y, temp_u, temp_p)],
                             ['y', 'u', 'p_dyn'], ['y_prime'] )
"""
def py_RK2_midpoint(y, u, p, dtau): # currently not in use
    # dtau is the variable of integration (dt or ds)
    k1      =  dtau * spatial_dynamics(y        , u, p)
    k2      =  dtau * spatial_dynamics(y + k1/2 , u, p)
    next_y  =  y  + k2
    return next_y
# construct casadi function, MX for discrete dynamics (integrator)
temp_y      = MX.sym('temp_y', c['N_S'])
temp_u      = MX.sym('temp_u', c['N_U'])
temp_p      = MX.sym('temp_p', c['N_PDYN'])
temp_dtau   = MX.sym('temp_dtau', 1)
discrete_dynamics = Function('discrete_dynamics', [temp_y, temp_u, temp_p, temp_dtau],
        [py_RK2_midpoint(temp_y, temp_u, temp_p, temp_dtau)],
        ['y', 'u', 'p_dyn', 'dtau'], ['next_y'])
"""
def py_trapezoidal(y, u, p, yn, un, pn, dtau):
    # dtau is the variable of integration (dt or ds)
    return y + 1/2*dtau*(spatial_dynamics(y,u,p) + spatial_dynamics(yn,un,pn))
temp_y      = MX.sym('temp_y', c['N_S'])
temp_yn     = MX.sym('temp_y', c['N_S'])
temp_u      = MX.sym('temp_u', c['N_U'])
temp_un     = MX.sym('temp_u', c['N_U'])
temp_p      = MX.sym('temp_p', c['N_PDYN'])
temp_pn     = MX.sym('temp_p', c['N_PDYN'])
temp_dtau   = MX.sym('temp_dtau', 1)
trapezoidal = Function( 'trapezoidal',
    [ temp_y,  temp_u,  temp_p, temp_yn, temp_un, temp_pn, temp_dtau ],
    [py_trapezoidal(temp_y,temp_u,temp_p,temp_yn,temp_un,temp_pn,temp_dtau)],
    ['y', 'u', 'p_dyn', 'yn', 'un', 'pn', 'dtau'], ['next_y'] )

def py_force_inequality(z, p_dyn):
    z = scale(z)
    del_dot      = z[ 0]                 # u
    fx_dot       = z[ 1]
    del_dot_c    = z[ 2]                 # u_c
    fx_dot_c     = z[ 3]
    delta        = z[ 4]                 # x
    fx           = z[ 5]
    r            = z[ 6]
    uy           = z[ 7]
    ux           = z[ 8]
    dpsi         = z[ 9]
    e            = z[10]
    t            = z[11]
    delta_c      = z[12]                 # x_c
    fx_c         = z[13]
    r_c          = z[14]
    uy_c         = z[15]
    ux_c         = z[16]
    dpsi_c       = z[17]
    e_c          = z[18]
    t_c          = z[19]

    k           = p_dyn[ 0]
    a           = p_dyn[ 1]
    b           = p_dyn[ 2]
    h           = p_dyn[ 3]
    m           = p_dyn[ 4]
    Izz         = p_dyn[ 5]
    Cf          = p_dyn[ 6]
    Cr          = p_dyn[ 7]
    mu          = p_dyn[ 8]
    mu_c        = p_dyn[ 9]
    Cd0_N       = p_dyn[10]
    Cd1_Npmps   = p_dyn[11]
    l           = a + b
    g           = 9.81

    fzf = m*b*g/l
    #fzr_N = ( m*a*g + h*2*fx ) / l

    # slip angles
    af   = atan2( uy  +a*r  , ux   ) - delta
    af_c = atan2( uy_c+a*r_c, ux_c ) - delta_c
    #ar = atan2( uy-b*r, ux )

    fxf       =  fx  *b/l # only for negative braking forces
    fxf_c     =  fx_c*b/l #   therefore min/max unnecesary
    fxf_min   = -cos(af)  *mu  *fzf
    fxf_min_c = -cos(af_c)*mu_c*fzf

    # fxf > fxf_min
    fx_lb   = ( fxf   - fxf_min   * c['max_fx_frac'] ) / c["scale_fx"]
    fx_lb_c = ( fxf_c - fxf_min_c * c['max_fx_frac'] ) / c["scale_fx"]

    return [ fx_lb, fx_lb_c ] # [fxf_lb, fxr_lb]
# construct casadi function, SX
temp_z = SX.sym( 'temp_z', c['N_S']+c['N_U'] )
temp_p = SX.sym( 'temp_z', c['N_PDYN'] )
force_inequality = Function( 'force_inequality', [temp_z, temp_p],
			     [vertcat(*py_force_inequality(temp_z, temp_p))],
                             ['z', 'p_dyn'], ['fx_bounds'] )
"""
# returns s,e coordinates of the front & rear circles representing the vehicle
def compute_veh_approx(s, e, dpsi, veh_a, veh_b, veh_rad):
    # vehicle approximation as one rectangle
    b_a = veh_a + 0.4953
    b_b = veh_b + 0.5715
    w = 1.87

    # distance from cg to the center of the vehicle
    offset = (b_a - b_b)/2

    # vehicle approximation as two circles
    veh_rad = 1.1
    veh_cg  = [s, e]
    R = [ [cos(dpsi), -sin(dpsi)],
          [sin(dpsi),  cos(dpsi)] ]
    #veh_f = mtimes(R, [ veh_rad + offset, 0]) + veh_cg
    #veh_r = mtimes(R, [-veh_rad + offset, 0]) + veh_cg
    veh_f = [ cos(dpsi)*( veh_rad + offset) + veh_cg[0],
              sin(dpsi)*( veh_rad + offset) + veh_cg[1] ]
    veh_r = [ cos(dpsi)*(-veh_rad + offset) + veh_cg[0],
              sin(dpsi)*(-veh_rad + offset) + veh_cg[1] ]
    return veh_f, veh_r, veh_rad

# calculates distance from vehicle to obstacle, assuming the obs is a circle
def obs_distance(s, e, dpsi, obs_s, obs_e, obs_r, veh_a, veh_b, veh_rad):
    veh_f, veh_r, veh_rad = compute_veh_approx( s,     e,     dpsi,
                                                veh_a, veh_b, veh_rad )
    dist1 = sqrt( (veh_f[0]-obs_s)**2 + (veh_f[1]-obs_e)**2) - obs_r - veh_rad
    dist2 = sqrt( (veh_r[0]-obs_s)**2 + (veh_r[1]-obs_e)**2) - obs_r - veh_rad
    return dist1, dist2

# calculates (unsigned) distance from vehicle to road edge
def edge_distance(s, e, dpsi, e_min, e_max, veh_a, veh_b, veh_rad):
    veh_f, veh_r, veh_rad = compute_veh_approx( s,     e,     dpsi,
                                                veh_a, veh_b, veh_rad )
    R1 = sqrt( (veh_f[1] - e_min)**2 ) - veh_rad
    R2 = sqrt( (veh_r[1] - e_min)**2 ) - veh_rad
    L1 = sqrt( (veh_f[1] - e_max)**2 ) - veh_rad
    L2 = sqrt( (veh_r[1] - e_max)**2 ) - veh_rad
    return R1, R2, L1, L2

def signed_distance_cost(target, dist):
    #cost = if_else( dist>target, 0, (target-dist)**2 )
    #cost = cost * if_else( dist<0, 2, 1 ) # put some comments here!
    cost = if_else( dist>target, 0, (target-dist)**2 )
    return cost
"""

def py_stagewise_cost(z,p_cost):
    z = scale(z)
    delta_dot   = z[ 0]                 # u
    fx_dot      = z[ 1]
    delta_dot_c = z[ 2]                 # u_c
    fx_dot_c    = z[ 3]
    delta       = z[ 4]                 # x
    fx          = z[ 5]
    r           = z[ 6]
    uy          = z[ 7]
    ux          = z[ 8]
    dpsi        = z[ 9]
    e           = z[10]
    t           = z[11]
    delta_c     = z[12]                 # x_c
    fx_c        = z[13]
    r_c         = z[14]
    uy_c        = z[15]
    ux_c        = z[16]
    dpsi_c      = z[17]
    e_c         = z[18]
    t_c         = z[19]

    s           = p_cost[ 0]
    e_des       = p_cost[ 1]
    ux_des      = p_cost[ 2]
    e_min       = p_cost[ 3]
    e_max       = p_cost[ 4]
    #obs_s       = p_cost[5]
    #obs_e       = p_cost[6]
    #obs_rad     = p_cost[7]
    Rd_delta    = p_cost[ 8]
    Rd_delta_c  = p_cost[ 9]
    Rd_fx       = p_cost[10]
    Rd_fx_c     = p_cost[11]
    R_fx        = p_cost[12]
    R_fx_c      = p_cost[13]
    Q_e         = p_cost[14]
    Q_e_c       = p_cost[15]
    Q_psi       = p_cost[16]
    Q_psi_c     = p_cost[17]
    Q_ux        = p_cost[18]
    Q_ux_c      = p_cost[19]
    Q_obs       = p_cost[20]
    edge_buffer = p_cost[21]
    veh_rad     = p_cost[22]

    # check size of p_cost
    if max(p_cost.size()) != c['N_PCOST']:
        raise ValueError( ('in `stagewise_cost`, size of p_cost ({}) does not'
                           'match N_PCOST ({})')
                          .format(max(p_cost.size()), c['N_PCOST']) )

    #fx_brake = fmin(fx, 0)
    #dis1, dis2 = obs_distance(s,e,dpsi,obs_s,obs_e,obs_rad,veh_a,veh_b,veh_rad)
    #R1,R2,L1,L2 = edge_distance(s,e,dpsi,e_min,e_max,veh_a,veh_b,veh_rad)
    #L   = fmax( 0, e   + veh_rad + edge_buffer - e_max )
    #R   = fmax( 0,-e   + veh_rad + edge_buffer + e_min )
    #L_c = fmax( 0, e_c + veh_rad + edge_buffer - e_max )
    #R_c = fmax( 0,-e_c + veh_rad + edge_buffer + e_min )
    #af   = atan2( uy   + veh_a*r   , ux   ) - delta
    af_c = atan2( uy_c + 1.2169*r_c , ux_c ) - delta_c
    ar_c = atan2( uy_c - 1.4131*r_c , ux_c )

    # nominal costs
    J  = Rd_delta     * delta_dot         **2
    J += Rd_fx        * fx_dot            **2
    J += R_fx         * fx                **2
    J += Q_e          * e                 **2
    J += Q_ux         * (ux - ux_des)     **2
    #J += Q_psi        * dpsi              **2
    #J += Q_obs        * ( L + R )         **2

    # contingency costs
    J += Rd_delta_c   * delta_dot_c       **2
    J += Rd_fx_c      * fx_dot_c          **2
    J += R_fx_c       * fx_c              **2
    J += Q_e_c        * e_c               **2
    J += Q_ux_c       * (ux_c - ux_des)   **2
    J += Q_psi_c      * (af_c**2 + ar_c**2)
    #J += Q_obs        * (L_c   + R_c    ) **2

    # coupled costs
    #J += R_delta /100   * (delta - delta_c) **2
    #J += R_fx    /100   * (fx    - fx_c   ) **2
    #J += Q_e     /100   * (e     - e_c    ) **2
    #J += Q_psi   /100   * (dpsi  - dpsi_c ) **2

    # obstacle costs
    #J += Q_obs * exp( e - e_max + veh_rad + edge_buffer )
    #J += Q_obs * exp(-e + e_min + veh_rad + edge_buffer )
    #J += Q_obs * signed_distance_cost(obs_buffer, dis etc)
    #J += Q_obs * signed_distance_cost(edge_buffer,L R etc)

    return J
# construct casadi function, SX for stagewise_cost
temp_z = SX.sym('temp_z', c['N_S']+c['N_U'])
temp_p = SX.sym('temp_p', c['N_PCOST'])
stagewise_cost = Function('stagewise_cost', [temp_z, temp_p],
        [py_stagewise_cost(temp_z, temp_p)], ['z', 'p_cost'], ['J_stage'])

################################################################################
# setup OCP

z   = []
p   = []

J   = 0
g   = []

z0  = []
lbw = []
ubw = []
lbg = []
ubg = []

# parameters, initial state
p_initx = MX.sym(   'p_initx',     c['N_S'] )
p.append(            p_initx                )

# parameters along prediction horizon
p_s        = MX.sym( 'p_s',        c['N'] )
p.append(             p_s                 )
p_k        = MX.sym( 'p_k',        c['N'] )
p.append(             p_k                 )
p_ux_des   = MX.sym( 'p_ux_des',   c['N'] )
p.append(             p_ux_des            )
p_e_des    = MX.sym( 'p_e_des',    c['N'] )
p.append(             p_e_des             )
p_e_min    = MX.sym( 'p_e_min',    c['N'] )
p.append(             p_e_min             )
p_e_max    = MX.sym( 'p_e_max',    c['N'] )
p.append(             p_e_max             )
p_obs1_s   = MX.sym( 'p_obs1_s',   c['N'] )
p.append(             p_obs1_s            )
p_obs1_e   = MX.sym( 'p_obs1_e',   c['N'] )
p.append(             p_obs1_e            )
p_obs1_rad = MX.sym( 'p_obs1_rad', c['N'] )
p.append(             p_obs1_rad          )
p_obs2_s   = MX.sym( 'p_obs2_s',   c['N'] )
p.append(             p_obs2_s            )
p_obs2_e   = MX.sym( 'p_obs2_e',   c['N'] )
p.append(             p_obs2_e            )
p_obs2_rad = MX.sym( 'p_obs2_rad', c['N'] )
p.append(             p_obs2_rad          )

# scalar parameters, vehicle model
p_a    = MX.sym( 'p_a',     1 )
p.append(         p_a         )
p_b    = MX.sym( 'p_b',     1 )
p.append(         p_b         )
p_h    = MX.sym( 'p_h',     1 )
p.append(         p_h         )
p_m    = MX.sym( 'p_m',     1 )
p.append(         p_m         )
p_I    = MX.sym( 'p_I',     1 )
p.append(         p_I         )
p_Cf   = MX.sym( 'p_Cf',    1 )
p.append(         p_Cf        )
p_Cr   = MX.sym( 'p_Cr',    1 )
p.append(         p_Cr        )
p_mu   = MX.sym( 'p_mu',    1 )
p.append(         p_mu        )
p_mu_c = MX.sym( 'p_mu_c',  1 )
p.append(         p_mu_c      )
p_Cd0  = MX.sym( 'p_Cd0',   1 )
p.append(         p_Cd0       )
p_Cd1  = MX.sym( 'p_Cd1',   1 )
p.append(         p_Cd1       )

# scalar parameters, cost function
p_Rd_delta    = MX.sym( 'p_Rd_delta',    1 )
p.append(                p_Rd_delta        )
p_Rd_delta_c  = MX.sym( 'p_Rd_delta_c',  1 )
p.append(                p_Rd_delta_c      )
p_Rd_fx       = MX.sym( 'p_Rd_fx',       1 )
p.append(                p_Rd_fx           )
p_Rd_fx_c     = MX.sym( 'p_Rd_fx_c',     1 )
p.append(                p_Rd_fx_c         )
p_R_fx        = MX.sym( 'p_R_fx',        1 )
p.append(                p_R_fx            )
p_R_fx_c      = MX.sym( 'p_R_fx_c',      1 )
p.append(                p_R_fx_c          )
p_Q_e         = MX.sym( 'p_Q_e',         1 )
p.append(                p_Q_e             )
p_Q_e_c       = MX.sym( 'p_Q_e_c',       1 )
p.append(                p_Q_e_c           )
p_Q_psi       = MX.sym( 'p_Q_psi',       1 )
p.append(                p_Q_psi           )
p_Q_psi_c     = MX.sym( 'p_Q_psi_c',     1 )
p.append(                p_Q_psi_c         )
p_Q_ux        = MX.sym( 'p_Q_ux',        1 )
p.append(                p_Q_ux            )
p_Q_ux_c      = MX.sym( 'p_Q_ux_c',      1 )
p.append(                p_Q_ux_c          )
p_Q_obs       = MX.sym( 'p_Q_obs',       1 )
p.append(                p_Q_obs           )
p_obs_buffer  = MX.sym( 'p_obs_buffer',  1 )
p.append(                p_obs_buffer      )
p_edge_buffer = MX.sym( 'p_edge_buffer', 1 )
p.append(                p_edge_buffer     )
p_veh_rad     = MX.sym( 'p_veh_rad',     1 )
p.append(                p_veh_rad         )

# initial state constraint
Z_0 = MX.sym('Z_0', c['N_S']+c['N_U'])
z.append(Z_0)
g.append(p_initx - Z_0[4:20])

# e bound constraint
g.append(Z_0[10])
g.append(Z_0[18])

# initial dynamic fx constraints
p_dyn  = vertcat( p_k[0], p_a,    p_b,    p_h,    p_m,   p_I,
                  p_Cf,   p_Cr,   p_mu,   p_mu_c, p_Cd0, p_Cd1 )
fx_pos = force_inequality( Z_0, p_dyn )
g.append(fx_pos[0]) # fxf
g.append(fx_pos[1]) # fxf_c

# intial command constraints
g.append(Z_0[0] - Z_0[2]) # delta_dot[0] - delta_dot_c[0] = 0
g.append(Z_0[1] - Z_0[3]) # fx_dot[0]    - fx_dot_c[0]    = 0

# initial cost
p_cost = vertcat( p_s[0],      p_e_des[0],    p_ux_des[0],   p_e_min[0],
                  p_e_max[0],  p_obs1_s[0],   p_obs1_e[0],   p_obs1_rad[0],
                  p_Rd_delta,  p_Rd_delta_c,  p_Rd_fx,       p_Rd_fx_c,
                  p_R_fx,      p_R_fx_c,      p_Q_e,         p_Q_e_c,
                  p_Q_psi,     p_Q_psi_c,     p_Q_ux,        p_Q_ux_c,
                  p_Q_obs,     p_edge_buffer, p_veh_rad )
J += stagewise_cost(Z_0, p_cost)

# stage costs and constraints
prev_Z = Z_0
for i in range(1,c['N']):
    Zi = MX.sym('Z_'+str(i), c['N_S']+c['N_U'])
    z.append(Zi)

    # dynamics
    if i <= c["N_REPLAN"] - 1: # use mu measured / nominal
        p_dyn  = vertcat( p_k[i-1], p_a,  p_b,  p_h,    p_m,   p_I,
                          p_Cf,     p_Cr, p_mu, p_mu,   p_Cd0, p_Cd1 )
        pn_dyn = vertcat( p_k[i],   p_a,  p_b,  p_h,    p_m,   p_I,
                          p_Cf,     p_Cr, p_mu, p_mu,   p_Cd0, p_Cd1 )
        ds     = p_s[i] - p_s[i-1]
        g.append(Zi[4:20] - trapezoidal(prev_Z[4:20], prev_Z[0:4], p_dyn,
                                        Zi[4:20],     Zi[0:4],     pn_dyn, ds))
    elif i <= c["N_REPLAN"]:   # transition to mu_c
        p_dyn  = vertcat( p_k[i-1], p_a,  p_b,  p_h,    p_m,   p_I,
                          p_Cf,     p_Cr, p_mu, p_mu,   p_Cd0, p_Cd1 )
        pn_dyn = vertcat( p_k[i],   p_a,  p_b,  p_h,    p_m,   p_I,
                          p_Cf,     p_Cr, p_mu, p_mu_c, p_Cd0, p_Cd1 )
        ds     = p_s[i] - p_s[i-1]
        g.append(Zi[4:20] - trapezoidal(prev_Z[4:20], prev_Z[0:4], p_dyn,
                                        Zi[4:20],     Zi[0:4],     pn_dyn, ds))
    else:                      # use mu_c
        p_dyn  = vertcat( p_k[i-1], p_a,  p_b,  p_h,    p_m,   p_I,
                          p_Cf,     p_Cr, p_mu, p_mu_c, p_Cd0, p_Cd1 )
        pn_dyn = vertcat( p_k[i],   p_a,  p_b,  p_h,    p_m,   p_I,
                          p_Cf,     p_Cr, p_mu, p_mu_c, p_Cd0, p_Cd1 )
        ds     = p_s[i] - p_s[i-1]
        g.append(Zi[4:20] - trapezoidal(prev_Z[4:20], prev_Z[0:4], p_dyn,
                                        Zi[4:20],     Zi[0:4],     pn_dyn, ds))
    # cost function
    p_cost = vertcat( p_s[i],      p_e_des[i],    p_ux_des[i],   p_e_min[i],
                      p_e_max[i],  p_obs1_s[i],   p_obs1_e[i],   p_obs1_rad[i],
                      p_Rd_delta,  p_Rd_delta_c,  p_Rd_fx,       p_Rd_fx_c,
                      p_R_fx,      p_R_fx_c,      p_Q_e,         p_Q_e_c,
                      p_Q_psi,     p_Q_psi_c,     p_Q_ux,        p_Q_ux_c,
                      p_Q_obs,     p_edge_buffer, p_veh_rad )
    J += stagewise_cost(Zi, p_cost)

    #if i == c['N']-1: # terminal costs & constraints
        #J += 10*p_Q_psi * ( fmax(Zi[17],0.2) + fmin(Zi[17],-0.2) )**2 # dPsi_c
        #J += 10*p_Q_e   * ( fmax(Zi[18],0.5) + fmin(Zi[18],-0.5) )**2 # e_c
        #J += 1  * p_Q_psi * Zi[17]**2 # dPsi_c
        #J += 10 * p_Q_e   * Zi[18]**2 # e_c

    # e bound constraint
    g.append(Zi[10])
    g.append(Zi[18])

    # dynamic fx constraints
    fx_pos = force_inequality(Zi, p_dyn)
    g.append(fx_pos[0]) #fxf
    g.append(fx_pos[1]) #fxf_c

    # command constraints
    g.append(Zi[0] - Zi[2]) # delta_dot[0] - delta_dot_c[0] = 0
    g.append(Zi[1] - Zi[3]) # fx_dot[0]    - fx_dot_c[0]    = 0

    prev_Z = Zi

# transform stuff
z = vertcat(*z)
g = vertcat(*g)
p = vertcat(*p)

print('Number of optimization variables: {}'.format(z.size()))
print('Number of constraints: {}'           .format(g.size()))
print('Number of parameters: {}'            .format(p.size()))

nlp      = {'x':z, 'p':p, 'g':g, 'f':J}
pysolver = nlpsol('pysolver', 'ipopt', nlp)

# generate code
print("generating c code... ", end='')
sys.stdout.flush()
pysolver.generate_dependencies("libcasadi_solver.c")
print("done")
print("compiling library... ", end='')
sys.stdout.flush()
os.system('gcc -fPIC -O3 -shared libcasadi_solver.c -o libcasadi_solver.so')
print("done")

