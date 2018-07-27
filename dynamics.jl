module Dynamics

using iLQR.Model, iLQR.Objective

export
    pendulum,
    doublependulum,
    cartpole,
    ballonbeam,
    acrobot

"""Simple Pendulum"""
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
model = Model(pendulum_dynamics,2,1)

n = model.n; # dimensions of system
m = model.m; # dimensions of control

# initial conditions
x0 = [0; 0];

# goal
xf = [pi; 0]; # (ie, swing up)

# costs
Q = 1e-3*eye(n);
Qf = 100*eye(n);
R = 1e-3*eye(m);

# simulation
tf = 5;

# Set up problem
obj = Objective(Q, R, Qf, tf, x0, xf)
pendulum = [model,obj]


"""Double Pendulum"""
# Load URDF
urdf_doublependulum = "urdf/doublependulum.urdf"
model = Model(urdf_doublependulum)
n = model.n; # dimensions of system
m = model.m; # dimensions of control

# initial and goal states
x0 = [0.;0.;0.;0.]
xf = [pi;0.;0.;0.]

# costs
Q = 0.0001*eye(4)
Qf = 250.0*eye(4)
R = 0.0001*eye(2)

# simulation
tf = 5.0

obj = Objective(Q,R,Qf,tf,x0,xf)
doublependulum = [model,obj]

"""## Dubins car"""
function dubins_dynamics(x,u)
    return [u[1]*cos(x[3]); u[1]*sin(x[3]); u[2]]
end

dubins = Model(dubins_dynamics,3,2)


"""Ball on beam"""
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



## Acrobot
acrobot = Model(urdf_doublependulum,[0.;1.])

## Cartpole / Inverted Pendulum
urdf_cartpole = "urdf/cartpole.urdf"
cartpole = Model(urdf_cartpole,[1.;0.])


end
