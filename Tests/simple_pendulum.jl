n = 2; # dimensions of system
p = 1; # dimensions of control
m = 1; # mass
l = 1; # length
g = 9.8; # gravity
mu = 0.01; # friction coefficient

dynamics(x,u) = [x[2];
        (g/l*sin(x[1]) - mu/m/(l^2)*x[2] + 1/m/(l^2)*u)];

# initial conditions
x0 = [0; 0];

# goal
xf = [pi; 0]; # (ie, swing up)

# costs
Q = 1e-3*eye(n);
Qf = 100*eye(n);
R = 1e-3*eye(p);

# simulation
dt = 0.1;
tf = 5;
N = Int(floor(tf/dt));
t = linspace(0,tf,N);

# Set up problem
model = iLQR.Model(dynamics, n, p)
obj = iLQR.Objective(Q, R, Qf, tf, x0, xf)
solver = iLQR.Solver(model, obj, dt=dt)