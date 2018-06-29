module Pendulum_v2
using iLQR
using Plots

n = 2; # dimensions of system
p = 1; # dimensions of control
m = 1; # mass
l = 1; # length
g = 9.8; # gravity
mu = 0.01; # friction coefficient

dt = 0.1
dynamics(x,u) = [x[2];
        (g/l*sin(x[1]) - mu/m/(l^2)*x[2] + 1/m/(l^2)*u)];

dynamics_midpoint = iLQR.f_midpoint(dynamics, dt)

fx(x,dt) = [1 + g*cos(x[1])*(dt^2)/(2*l) dt - mu*(dt^2)/(2*m*l^2);
            g*cos(x[1] + x[2]*dt/2)*dt/l - mu*g*cos(x[1])*(dt^2)/(2*m*l^3) 1 + g*cos(x[1] + x[2]*dt/2)*(dt^2)/(2*l) - mu*dt/(m*l^2) + (mu^2)*(dt^2)/(2*(m^2)*l^4)];

fu(x,dt) = [(dt^2)/(2*m*l^2);
            (-mu*(dt^2)/(2*(m^2)*l^4) + dt/(m*l^2))];


# initial conditions
x0 = [0; 0];

# goal
xf = [pi; 0]; # (ie, swing up)

# costs
Q = 1e-5*eye(n);
Qf = 25*eye(n);
R = 1e-5*eye(p);

e_dJ = 1e-6;

# simulation
dt = 0.1;
tf = 1;
N = Int(floor(tf/dt));
t = linspace(0,tf,N);
iterations = 4

# Set up problem
model = iLQR.Model(dynamics, n, p)
obj = iLQR.Objective(Q, R, Qf, tf, x0, xf)
solver = iLQR.Solver(model, obj, fx, fu, dt)

# initialization
u = zeros(p,N-1);
x = zeros(n,N);
x_ = similar(x)
u_ = similar(u)
x[:,1] = x0;

K = zeros(m,n,N)
lk = zeros(m,N)

# first roll-out
iLQR.rollout!(solver, x, u)

## iterations of iLQR using my derivation
# improvement criteria
c1 = 0.25;
c2 = 0.75;

for i = 1:iterations
    K, lk = iLQR.backward_pass!(solver, x, u, K, lk)
    J = iLQR.forwardpass!(solver, x, u, K, lk, x_, u_)

    x = copy(x_)
    u = copy(u_)
    println("Cost:", J)
end

p = plot(linspace(0,tf,N), x[1,:])
p = plot!(linspace(0,tf,N), x[2,:])
# display(p)


end
