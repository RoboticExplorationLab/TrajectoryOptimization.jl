## Ball on beam
# TODO needs inplace dynamics and objective
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
n,m = 4,1
model = Model(ballonbeam_dynamics,n,m)

# initial and goal states
x0 = [.1;0;0.;0.]
xf = [.5;0.;0.;0.]

# costs
#TODO these costs are taken from the notebook
Q = 5e-4*Diagonal(I,n)
Qf = 500.0*Diagonal(I,n)
R = 1e-5*Diagonal(I,m)

# simulation
tf = 1.0
dt = 0.01

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

ballonbeam = [model, obj_uncon]
