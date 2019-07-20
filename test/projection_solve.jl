
# Set up Problem
model = Dynamics.car
n,m = model.n, model.m
N = 51
Q = Diagonal(I,n)*0.01
R = Diagonal(I,m)*0.01
Qf = Diagonal(I,n)*0.01
xf = [0,1,0]
obj = LQRObjective(Q,R,Qf,xf,N)
n,m = model.n, model.m
bnd = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf], u_min=[0.1,-2], u_max=1.2)
bnd1 = BoundConstraint(n,m, u_min=bnd.u_min, u_max=bnd.u_max)
bnd_x = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf])
goal = goal_constraint(xf)
obs = (([0.2, 0.6], 0.25),
       ([-0.5, 0.5], 0.4),
       ([-0.15,0.85], 0.1))
obs1 = TO.planar_obstacle_constraint(n,m, obs[1]..., :obstacle1)
obs2 = TO.planar_obstacle_constraint(n,m, obs[2]..., :obstacle2)
obs3 = TO.planar_obstacle_constraint(n,m, obs[3]..., :obstacle3)
con = Constraints(N)
con[1] += bnd1
for k = 2:N-1
    con[k] += bnd  + obs1 + obs2 + obs3
end
con[N] += goal
prob = Problem(rk4(model), obj, constraints=con, tf=3)

# Solve with ALTRO
initial_controls!(prob, ones(m,N-1))
ilqr = iLQRSolverOptions()
al = AugmentedLagrangianSolverOptions(opts_uncon=ilqr)
al.penalty_initial = 0.001
al.constraint_tolerance = 1e-2
al.constraint_tolerance_intermediate = 1e-2
al.verbose = true
res, = solve(prob, al)
max_violation(res)


plot()
TO.plot_circle!(obs[1]...)
TO.plot_circle!(obs[2]...)
TO.plot_circle!(obs[3]...)
plot_trajectory!(res.X, markershape=:circle)
plot(res.U)

# Projected Newton
opts = ProjectedNewtonSolverOptions{Float64}(verbose=true)
solver = ProjectedNewtonSolver(res,opts)
TO.update!(res, solver)
V0 = copy(solver.V.V)

Z = TO.primals(solver.V)
a = solver.a.duals
Y,y = TO.active_constraints(prob, solver)
viol0 = norm(y,Inf)
HinvY = Diagonal(solver.H)\Y'
S = cholesky(Symmetric(Y*HinvY))

solver.opts.feasibility_tolerance = 1e-12
solver.opts.verbose = true
solver.opts.active_set_tolerance = 1e-6
begin
    copyto!(solver.V.V, V0)
    TO.update!(res, solver)
    TO.projection_solve!(prob, solver)
end
copyto!(solver.V.V, V0)
TO.update!(res, solver)
TO._projection_solve!(prob, solver, solver.V, true)
TO._projection_linesearch!(prob, solver, solver.V, S, Hinv*Y')

dZ = -Hinv*Y'*(S\y)
Z .+= dZ

TO.dynamics_constraints!(prob, solver)
TO.update_constraints!(prob, solver)
y = solver.y[a]
viol = norm(y,Inf)

TO.projection!(res, solver)

V_ = TO.newton_step!(res, solver)

plot!(solver.V.U)

A = rand(10,10)
B = A + 1e-4*I
b = rand(10)



x = reg_solve(A,b,1e-2)
norm(A*x-b)
