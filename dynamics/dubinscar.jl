## Dubins car
# TODO test and inplace dynamics

function dubins_dynamics!(xdot,x,u)
    xdot[1] = u[1]*cos(x[3])
    xdot[2] = u[1]*sin(x[3])
    xdot[3] = u[2]
    xdot
end

function dubins_dynamics(x,u)
    return [u[1]*cos(x[3]); u[1]*sin(x[3]); u[2]]
end

model = Model(dubins_dynamics,3,2)
model! = Model(dubins_dynamics!,3,2)


# initial and goal states
x0 = [0.;0.;0.]
xf = [0.;1.;0.]

# costs
Q = 0.001*eye(model.n)
Qf = 1000.0*eye(model.n)
R = 0.001*eye(model.m)

# simulation
tf = 5.0
dt = 0.01

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

dubinscar = [model, obj_uncon]
dubinscar! = [model!, obj_uncon]
