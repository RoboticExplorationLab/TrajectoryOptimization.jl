include("model.jl")

export
    pendulum,
    doublependulum,
    cartpole,
    ballonbeam,
    acrobot

## Simple Pendulum
# https://github.com/HarvardAgileRoboticsLab/unscented-dynamic-programming/blob/master/pendulum_dynamics.m
function pendulum_dynamics(x,u)
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    return [x[2]; (u - m*g*lc*sin(x[1]) - b*x[2])];
end
pendulum = Model(pendulum_dynamics,2,1)

## Dubins car
function dubins_dynamics(x,u)
    return [u[1]*cos(x[3]); u[1]*sin(x[3]); u[2]]
end

dubins = Model(dubins_dynamics,3,2)

## Ball on beam
function ballonbeam_dynamics(x,u)
    g = 9.81
    m1 = .35
    m2 = 2
    l = 0.5

    z = x[1];
    theta = x[2];
    zdot = x[3];
    thetadot = x[4];
    F = u[1];
    zddot = z*thetadot^2-g*sin(theta);
    thetaddot = (F*l*cos(theta) - 2*m1*z*zdot*thetadot - m1*g*z*cos(theta) - (m2*g*l*cos(theta))/2) / (m2*l^2/3 + m1*z^2);

    sys = [zdot; thetadot; zddot; thetaddot];
end

ballonbeam = Model(ballonbeam_dynamics,4,1)

## Double Pendulum
urdf_doublependulum = "urdf/doublependulum.urdf"
doublependulum = Model(urdf_doublependulum)

## Acrobot
doublependulum = Model(urdf_doublependulum,[0.;1.])

## Cartpole / Inverted Pendulum
urdf_cartpole = "urdf/cartpole.urdf"
cartpole = Model(urdf_cartpole,[1.;0.])
