## Pendulum
# https://github.com/HarvardAgileRoboticsLab/unscented-dynamic-programming/blob/master/pendulum_dynamics.m
function pendulum_dynamics!(xdot,x,u)
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    xdot[1] = x[2]
    xdot[2] = (u[1] - m*g*lc*sin(x[1]) - b*x[2])/I
end
n,m = 2,1
pendulum_model = Model(pendulum_dynamics!,n,m) # inplace model

# initial conditions
x0 = [0; 0.]

# goal
xf = [pi; 0] # (ie, swing up)

# costs
Q = 1e-3*Matrix(I,n,n)
Qf = 100.0*Matrix(I,n,n)
R = 1e-2*Matrix(I,m,m)

# simulation
tf = 5.

# Create the cost
pendulum_cost = LQRCost(Q, R, Qf, xf)

# Constraints
u_bound = 2
bnd = bound_constraint(n, m, u_min=-u_bound, u_max=u_bound)

#Create Problem
N = 51
pendulum_new = Problem(pendulum_model, pendulum_cost, constraints=[bnd], N=N, tf=tf)
initial_controls!(pendulum_new, rand(m, N-1))

# Set up (old) problem
obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
pendulum = [pendulum_model,obj_uncon]

obj_con = ConstrainedObjective(obj_uncon, u_min=-u_bound, u_max=u_bound) # constrained objective
pendulum_constrained = [pendulum_model, obj_con]
