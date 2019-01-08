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

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

solver = TrajectoryOptimization.Solver(model,obj,dt=dt,integration=:rk3)
U = zeros(m,solver.N-1)

results, stats = solve(solver,U)

plot(to_array(results.X)')
plot(to_array(results.X)[1,:],to_array(results.X)[2,:],label="",title="Parallel Park",xlabel="x",ylabel="y")
results.X[end]
