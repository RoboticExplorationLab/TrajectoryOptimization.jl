using LinearAlgebra
## Dubins car
# TODO test and inplace dynamics

function dubins_dynamics(x,u)
    return [u[1]*cos(x[3]); u[1]*sin(x[3]); u[2]]
end

model = Model(dubins_dynamics,3,2)

# initial and goal states
x0 = [0.;0.;0.]
xf = [0.;1.;0.]

# costs
Q = 0.001*Diagonal{Float64}(I, model.n)
Qf = 1000.0*Diagonal{Float64}(I, model.n)
R = 0.001*Diagonal{Float64}(I, model.m)

# simulation
tf = 5.0
dt = 0.01

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

dubinscar = [model, obj_uncon]
