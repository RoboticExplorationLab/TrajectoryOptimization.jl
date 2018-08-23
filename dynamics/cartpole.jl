## Cartpole / Inverted Pendulum
#TODO test
urdf_folder = joinpath(Pkg.dir("TrajectoryOptimization"), "dynamics/urdf")
urdf_cartpole = joinpath(urdf_folder, "cartpole.urdf")

model = Model(urdf_cartpole,[1.;0.]) # underactuated, only control of slider

M = 2;  # kg
mm = 0.5;  # kg
ell = 0.5;  # m
I = 0.006;  # kg m^2
b = 0.1;  # friction N/(m/s)
g = 9.81;  # gravity m/s/s


function cartpole_dynamics!(Xdot, X, U)
    mc = 10;   # mass of the cart in kg
    mp = 1;    # mass of the pole (point mass at the end) in kg
    l = 0.5;   # length of the pole in m
    g = 9.81;  # gravity m/s^2

    q = X[1:2];
    qd = X[3:4];

    s = sin(q[2])
    c = cos(q[2])

    H = [mc+mp mp*l*c; mp*l*c mp*l^2];
    C = [0 -mp*qd[2]*l*s; 0 0];
    G = [0; mp*g*l*s];
    B = [1; 0];

    qdd = -H\(C*qd + G - B*U');

    Xdot[1:2] = qd
    Xdot[3:4] = qdd
    return nothing
end

# initial and goal states
x0 = [0.;pi;0.;0.]
xf = [0.;0.;0.;0.]

# costs
Q = 0.001*eye(model.n)
Qf = 150.0*eye(model.n)
R = 0.0001*eye(model.m)

# simulation
tf = 5.0
dt = 0.1

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

cartpole = [model, obj_uncon]
