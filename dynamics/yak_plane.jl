using Parameters
using StaticArrays

@with_kw mutable struct YakPlane{R,T} <: RigidBody{R}
    g::T = 9.81; #Gravitational acceleration (m/s^2)
    rho::T = 1.2; #Air density at 20C (kg/m^3)
    m::T = .075; #Mass of plane (kg)

    Jx::T = 4.8944e-04; #roll axis inertia (kg*m^2)
    Jy::T = 6.3778e-04; #pitch axis inertia (kg*m^2)
    Jz::T = 7.9509e-04; #yaw axis inertia (kg*m^2)

    J::Diagonal{T,SVector{3,T}} = Diagonal(@SVector [Jx,Jy,Jz]);
    Jinv::Diagonal{T,SVector{3,T}} = Diagonal(@SVector [1/Jx,1/Jy,1/Jz]); #Assuming products of inertia are small

    Jm::T = .007*(.0075)^2 + .002*(.14)^2/12; #motor + prop inertia (kg*m^2)

    # All lifting surfaces are modeled as unsweapt tapered wings
    b::T = 45/100; #wing span (m)
    l_in::T = 6/100; #inboard wing length covered by propwash (m)
    cr::T = 13.5/100; #root chord (m)
    ct::T = 8/100; #tip chord (m)
    cm::T = (ct + cr)/2; #mean wing chord (m)
    S::T = b*cm; #planform area of wing (m^2)
    S_in::T = 2*l_in*cr;
    S_out::T = S-S_in;
    #Ra::T = b^2/S; %wing aspect ratio (dimensionless)
    Rt::T = ct/cr; #wing taper ratio (dimensionless)
    r_ail::T = (b/6)*(1+2*Rt)/(1+Rt); #aileron moment arm (m)

    ep_ail::T = 0.63; #flap effectiveness (Phillips P.41)
    trim_ail::T = 106; #control input for zero deflection
    g_ail::T = (15*pi/180)/100; #maps control input to deflection angle

    b_elev::T = 16/100; #elevator span (m)
    cr_elev::T = 6/100; #elevator root chord (m)
    ct_elev::T = 4/100; #elevator tip chord (m)
    cm_elev::T = (ct_elev + cr_elev)/2; #mean elevator chord (m)
    S_elev::T = b_elev*cm_elev; #planform area of elevator (m^2)
    Ra_elev::T = b_elev^2/S_elev; #wing aspect ratio (dimensionless)
    r_elev::T = 22/100; #elevator moment arm (m)

    ep_elev::T = 0.88; #flap effectiveness (Phillips P.41)
    trim_elev::T = 106; #control input for zero deflection
    g_elev::T = (20*pi/180)/100; #maps control input to deflection angle

    b_rud::T = 10.5/100; #rudder span (m)
    cr_rud::T = 7/100; #rudder root chord (m)
    ct_rud::T = 3.5/100; #rudder tip chord (m)
    cm_rud::T = (ct_rud + cr_rud)/2; #mean rudder chord (m)
    S_rud::T = b_rud*cm_rud; #planform area of rudder (m^2)
    Ra_rud::T = b_rud^2/S_rud; #wing aspect ratio (dimensionless)
    r_rud::T = 24/100; #rudder moment arm (m)
    z_rud::T = 2/100; #height of rudder center of pressure (m)

    ep_rud::T = 0.76; #flap effectiveness (Phillips P.41)
    trim_rud::T = 106; #control input for zero deflection
    g_rud::T = (35*pi/180)/100; #maps from control input to deflection angle

    trim_thr::T = 24; #control input for zero thrust (deadband)
    g_thr::T = 0.006763; #maps control input to Newtons of thrust
    g_mot::T = 3000*2*pi/60*7/255; #maps control input to motor rad/sec
end

Base.size(::YakPlane) = 12,4

YakPlane() = YakPlane{MRP{Float64},Float64}()


function TrajectoryOptimization.dynamics(p::YakPlane, x::SVector, u::SVector)
    r,q,v,w = parse_state(p, x)

    Q = rotmat(q)

    # control input
    thr  = u[1]; #Throttle command (0-255 as sent to RC controller)
    ail  = u[2]; #Aileron command (0-255 as sent to RC controller)
    elev = u[3]; #Elevator command (0-255 as sent to RC controller)
    rud  = u[4]; #Rudder command (0-255 as sent to RC controller)

    #Note that body coordinate frame is:
    # x: points forward out nose
    # y: points out right wing tip
    # z: points down

    # ------- Input Checks -------- #
    thr  = clamp(thr,  0, 255)
    ail  = clamp(ail,  0, 255)
    elev = clamp(elev, 0, 255)
    rud  = clamp(rud,  0, 255)

    # ---------- Map Control Inputs to Angles ---------- #
    delta_ail = (ail-p.trim_ail)*p.g_ail;
    delta_elev = (elev-p.trim_elev)*p.g_elev;
    delta_rud = (rud-p.trim_rud)*p.g_rud;

    # ---------- Aerodynamic Forces (body frame) ---------- #
    v_body = Q'*v; #body-frame velocity
    v_rout = v_body + cross(w, @SVector [0,  p.r_ail, 0]);
    v_lout = v_body + cross(w, @SVector [0, -p.r_ail, 0]);
    v_rin  = v_body + cross(w, @SVector [0,  p.l_in, 0]) + propwash(thr);
    v_lin  = v_body + cross(w, @SVector [0, -p.l_in, 0]) + propwash(thr);
    v_elev = v_body + cross(w, @SVector [-p.r_elev, 0, 0]) + propwash(thr);
    v_rud  = v_body + cross(w, @SVector [-p.r_rud,  0, -p.z_rud]) + propwash(thr);

    # --- Outboard Wing Sections --- #
    a_rout = alpha(v_rout);
    a_lout = alpha(v_lout);
    a_eff_rout = a_rout + p.ep_ail*delta_ail; #effective angle of attack
    a_eff_lout = a_lout - p.ep_ail*delta_ail; #effective angle of attack

    F_rout = -p_dyn(p, v_rout)*.5*p.S_out* @SVector [Cd_wing(a_eff_rout), 0, Cl_wing(a_eff_rout)];
    F_lout = -p_dyn(p, v_lout)*.5*p.S_out* @SVector [Cd_wing(a_eff_lout), 0, Cl_wing(a_eff_lout)];

    F_rout = arotate(a_rout,F_rout); #rotate to body frame
    F_lout = arotate(a_lout,F_lout); #rotate to body frame

    # --- Inboard Wing Sections (Includes Propwash) --- #
    a_rin = alpha(v_rin);
    a_lin = alpha(v_lin);
    a_eff_rin = a_rin + p.ep_ail*delta_ail; #effective angle of attack
    a_eff_lin = a_lin - p.ep_ail*delta_ail; #effective angle of attack

    F_rin = -p_dyn(p, v_rin)*.5*p.S_in* @SVector [Cd_wing(a_eff_rin), 0, Cl_wing(a_eff_rin)];
    F_lin = -p_dyn(p, v_lin)*.5*p.S_in* @SVector [Cd_wing(a_eff_lin), 0, Cl_wing(a_eff_lin)];

    F_rin = arotate(a_rin,F_rin); #rotate to body frame
    F_lin = arotate(a_lin,F_lin); #rotate to body frame

    # --- Elevator --- #
    a_elev = alpha(v_elev);
    a_eff_elev = a_elev + p.ep_elev*delta_elev; #effective angle of attack

    F_elev = -p_dyn(p, v_elev)*p.S_elev*[Cd_elev(p, a_eff_elev), 0, Cl_plate(a_eff_elev)];

    F_elev = arotate(a_elev,F_elev); #rotate to body frame

    # --- Rudder --- #
    a_rud = beta(v_rud);
    a_eff_rud = a_rud - p.ep_rud*delta_rud; #effective angle of attack

    F_rud = -p_dyn(p, v_rud)*p.S_rud*[Cd_rud(p,a_eff_rud), Cl_plate(a_eff_rud), 0];

    F_rud = brotate(a_rud,F_rud); #rotate to body frame

    # --- Thrust --- #
    if thr > p.trim_thr
        F_thr = @SVector [(thr-p.trim_thr)*p.g_thr, 0, 0];
        w_mot = @SVector [p.g_mot*thr, 0, 0];
    else #deadband
        F_thr = @SVector zeros(3);
        w_mot = @SVector zeros(3);
    end

    # ---------- Aerodynamic Torques (body frame) ---------- #
    T_rout = cross([0; p.r_ail; 0],F_rout);
    T_lout = cross([0; -p.r_ail; 0],F_lout);

    T_rin = cross([0; p.l_in; 0],F_rin);
    T_lin = cross([0; -p.l_in; 0],F_lin);

    T_elev = cross([-p.r_elev; 0; 0],F_elev);

    T_rud = cross([-p.r_rud; 0; -p.z_rud],F_rud);

    # ---------- Add Everything Together ---------- #
    # problems: F_lout, F_rin
    F_aero = F_rout + F_lout + F_rin + F_lin + F_elev + F_rud + F_thr;
    F = Q*F_aero - @SVector [0, 0, p.m*p.g];

    T = T_rout + T_lout + T_rin + T_lin + T_elev + T_rud + cross((p.J*w + p.Jm*w_mot),w);

    rdot = v;
    qdot = kinematics(q, w)
    vdot = F/p.m
    wdot = p.Jinv*T

    # xdot = [v;
    #         .25*((1-r'*r)*w - 2*cross(w,r) + 2*(w'*r)*r);
    #         F/p.m;
    #         p.Jinv*T];
    #
    build_state(p, rdot, qdot, vdot, wdot)
end

"Angle of attack"
@inline alpha(v) = atan(v[3],v[1])

"Sideslip angle"
@inline beta(v) = atan(v[2],v[1])

"Rotate by angle of attack"
function arotate(a,r)
    sa,ca = sincos(a)
    R = @SMatrix [
        ca 0 -sa;
        0  1   0;
        sa 0  ca]
    R*r
end

"Rotate by sideslip angle"
function brotate(b,r)
    sb,cb = sincos(b)
    R = @SMatrix [
        cb -sb 0;
        sb  cb 0;
        0    0 1]
    R*r
end

""" Propwash wind speed (body frame)
Fit from anemometer data taken at tail
No significant different between wing/tail measurements
"""
function propwash(thr)::SVector{3}
    trim_thr = 24; # control input for zero thrust (deadband)
    if thr > trim_thr
        v = @SVector [5.568*thr^0.199 - 8.859, 0, 0];
    else #deadband
        v = @SVector zeros(3)
    end
end

"Dynamic pressure"
function p_dyn(p::YakPlane, v)
    pd = 0.5*p.rho*(v'v)
end

""" Lift coefficient (alpha in radians)
3rd order polynomial fit to glide-test data
Good to about ±20°
"""
function Cl_wing(a)
    a = clamp(a, -0.5*pi, 0.5*pi)
    cl = -27.52*a^3 - .6353*a^2 + 6.089*a;
end

""" Lift coefficient (alpha in radians)
Ideal flat plate model used for wing and rudder
"""
function Cl_plate(a)
    a = clamp(a, -0.5*pi, 0.5*pi)
    cl = 2pi*a
end

""" Drag coefficient (alpha in radians)
2nd order polynomial fit to glide-test data
Good to about ±20°
"""
function Cd_wing(a)
    a = clamp(a, -0.5*pi, 0.5*pi)
    cd = 2.08*a^2 + .0612;
end

""" Drag coefficient (alpha in radians)
Induced drag for a tapered finite wing
    From phillips P.55
"""
function Cd_elev(p::YakPlane, a)
    a = clamp(a, -0.5*pi, 0.5*pi)
    cd = (4*pi*a^2)/p.Ra_elev
end

""" Drag coefficient (alpha in radians)
Induced drag for a tapered finite wing
From Phillips P.55
"""
function Cd_rud(p::YakPlane, a)
    a = clamp(a, -0.5*pi, 0.5*pi)
    cd = (4*pi*a^2)/p.Ra_rud;
end
