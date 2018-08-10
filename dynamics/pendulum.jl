## Pendulum
# https://github.com/HarvardAgileRoboticsLab/unscented-dynamic-programming/blob/master/pendulum_dynamics.m

function pendulum_dynamics(x,u)
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    xdot = zeros(x)
    xdot[1] = x[2]
    xdot[2] = (u[1] - m*g*lc*sin(x[1]) - b*x[2])/I
    return xdot
end

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

model = Model(pendulum_dynamics,2,1)
model! = Model(pendulum_dynamics!,2,1) # inplace model

# initial conditions
x0 = [0; 0]

# goal
xf = [pi; 0] # (ie, swing up)

# costs
Q = (1e-3)*eye(model.n)
Qf = 100.0*eye(model.n)
R = (1e-2)*eye(model.m)

# simulation
tf = 5.

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

# Constraints
u_min = [-2]
u_max = [2]
x_min = [-5;-5]
x_max = [10; 10]
obj_con = ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

# Set up problem
pendulum = [model,obj_uncon]
pendulum! = [model!,obj_uncon]
pendulum_constrained = [model, obj_con]
pendulum_constrained! = [model!,obj_con]
