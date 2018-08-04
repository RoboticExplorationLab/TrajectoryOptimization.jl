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

fx(x,u,dt) = [1 + g*cos(x[1])*(dt^2)/(2*l) dt - mu*(dt^2)/(2*m*l^2);
            g*cos(x[1] + x[2]*dt/2)*dt/l - mu*g*cos(x[1])*(dt^2)/(2*m*l^3) 1 + g*cos(x[1] + x[2]*dt/2)*(dt^2)/(2*l) - mu*dt/(m*l^2) + (mu^2)*(dt^2)/(2*(m^2)*l^4)];

fu(x,u,dt) = [(dt^2)/(2*m*l^2);
            (-mu*(dt^2)/(2*(m^2)*l^4) + dt/(m*l^2))];

function f_jacobian(x,u,dt)
    A = [1 + g*cos(x[1])*(dt^2)/(2*l) dt - mu*(dt^2)/(2*m*l^2);
        g*cos(x[1] + x[2]*dt/2)*dt/l - mu*g*cos(x[1])*(dt^2)/(2*m*l^3) 1 + g*cos(x[1] + x[2]*dt/2)*(dt^2)/(2*l) - mu*dt/(m*l^2) + (mu^2)*(dt^2)/(2*(m^2)*l^4)];
    B = [(dt^2)/(2*m*l^2);
        (-mu*(dt^2)/(2*(m^2)*l^4) + dt/(m*l^2))];
    return A, B
end

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
# solver = iLQR.Solver(model, obj, fx, fu, dt) # initialization
solver = iLQR.Solver(model, obj, f_jacobian, dt) # initialization
x, u = iLQR.solve(solver, iterations=1)

p = plot(linspace(0,tf,N), x[1,:])
p = plot!(linspace(0,tf,N), x[2,:])
# display(p)


end
