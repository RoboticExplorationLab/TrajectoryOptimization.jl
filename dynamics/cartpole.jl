## Cartpole / Inverted Pendulum
#TODO test
import TrajectoryOptimization
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf_cartpole = joinpath(urdf_folder, "cartpole.urdf")

model_urdf = Model(urdf_cartpole,[1.;0.]) # underactuated, only control of slider

function cartpole_dynamics!(Xdot, X, U)
    mc = 2;   # mass of the cart in kg (10)
    mp = 0.5;    # mass of the pole (point mass at the end) in kg
    l = 0.5;   # length of the pole in m
    g = 9.81;  # gravity m/s^2

    q = X[1:2];
    qd = X[3:4];

    if isfinite(q[2])
        s = sin(q[2])
        c = cos(q[2])
    else
        s = Inf
        c = Inf
    end

    H = [mc+mp mp*l*c; mp*l*c mp*l^2];
    C = [0 -mp*qd[2]*l*s; 0 0];
    G = [0; mp*g*l*s];
    B = [1; 0];

    qdd = -H\(C*qd + G - B*U');

    Xdot[1:2] = qd
    Xdot[3:4] = qdd
    return nothing
end

function cartpole_dynamics_udp!(Xdot, X, U)
    mc = 10.0;   # mass of the cart in kg (10)
    mp = 1.0;    # mass of the pole (point mass at the end) in kg
    l = 0.5;   # length of the pole in m
    g = 9.81;  # gravity m/s^2

    q = X[1:2];
    qd = X[3:4];

    if isfinite(q[2])
        s = sin(q[2])
        c = cos(q[2])
    else
        s = Inf
        c = Inf
    end

    H = [mc+mp mp*l*c; mp*l*c mp*l^2];
    C = [0 -mp*qd[2]*l*s; 0 0];
    G = [0; mp*g*l*s];
    B = [1; 0];

    qdd = -H\(C*qd + G - B*U');

    Xdot[1:2] = qd
    Xdot[3:4] = qdd
    return nothing
end

n,m = 4,1
model_analytical = Model(cartpole_dynamics!,n,m)
model_udp = Model(cartpole_dynamics_udp!,n,m)

# initial and goal states
x0 = [0.;0.;0.;0.]
xf = [0.;pi;0.;0.]

x0_flipped = [0.;pi;0.;0.]
xf_flipped = [0.;0.;0.;0.]

# costs
Q = 0.01*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 0.01*Diagonal(I,m)

# simulation
tf = 5.0
dt = 0.1

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
obj_uncon_flipped = LQRObjective(Q, R, Qf, tf, x0_flipped, xf_flipped)

cartpole = [model_urdf, obj_uncon_flipped]
cartpole_analytical = [model_analytical, obj_uncon]
cartpole_udp = [model_udp, obj_uncon]
cartpole_mechanism = MechanismState(parse_urdf(Float64,urdf_cartpole))
