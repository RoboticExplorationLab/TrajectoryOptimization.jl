function double_integrator_dynamics!(ẋ,x,u)
    ẋ[1] = x[2]
    ẋ[2] = u[1]
end

n = 2
m = 1

model = Model(double_integrator_dynamics!,n,m)

# initial and goal states
x0 = [1.;0.]
xf = [0.;0.]

# costs
Q = (1e-2)*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

# simulation
tf = 5.0
dt = 0.1

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

double_integrator = [model, obj_uncon]
