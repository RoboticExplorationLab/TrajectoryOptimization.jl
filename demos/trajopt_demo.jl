function dubins_dynamics!(xdot,x,u)
    xdot[1] = u[1]*cos(x[3])
    xdot[2] = u[1]*sin(x[3])
    xdot[3] = u[2]
    xdot
end
n = 3
m = 2

model = Model(dubins_dynamics!,n,m)

x0 = [0.;0.;0.]
xf = [1.;1.;0.]

Q = Array((1e-2)*Diagonal(I,n))
Qf = Array(100.0*Diagonal(I,n))
R = Array((1e-2)*Diagonal(I,m))

tf = 5.0
dt = 0.01

obj = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

solver = TrajectoryOptimization.Solver(model,obj,dt=dt)
U = zeros(m,solver.N)

results, stats = solve(solver,U)

plot(to_array(results.X)')

results.X[end]
