## Dubins car
# TODO test and inplace dynamics

function dubins_dynamics!(xdot,x,u)
    xdot[1] = u[1]*cos(x[3])
    xdot[2] = u[1]*sin(x[3])
    xdot[3] = u[2]
    xdot
end
n,m = 3,2

model = Model(dubins_dynamics!,n,m)


# initial and goal states
x0 = [0.;0.;0.]
xf = [0.;1.;0.]

# costs
Q = (1e-2)*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

# simulation
tf = 5.0
dt = 0.01

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

dubinscar = [model, obj_uncon]

x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]

obj_con_box = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max)

dubinscar_parallelpark = [model, obj_con_box]
