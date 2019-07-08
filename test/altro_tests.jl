const TO = TrajectoryOptimization

# Set up Problem
model = Dynamics.car_model
costfun = Dynamics.car_costfun
xf = [0,1,0]
N = 51
n,m = model.n, model.m
bnd = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf], u_min=[0.1,-2], u_max=1.5)
bnd1 = BoundConstraint(n,m, u_min=bnd.u_min)
bnd_x = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf])
goal = goal_constraint(xf)
obs = (([0.2, 0.6], 0.25),
       ([-0.5, 0.5], 0.4))
obs1 = TO.planar_obstacle_constraint(n,m, obs[1]..., :obstacle1)
obs2 = TO.planar_obstacle_constraint(n,m, obs[2]..., :obstacle2)
con = Constraints(N)
con[1] += bnd1
for k = 2:N-1
    con[k] += bnd  + obs1 + obs2
end
con[N] += goal
prob = Problem(rk4(model), Objective(costfun, N), constraints=con, tf=3)

# Solve with ALTRO
initial_controls!(prob, ones(m,N-1))
ilqr = iLQRSolverOptions()
al = AugmentedLagrangianSolverOptions(opts_uncon=ilqr)
al.constraint_tolerance = 1e-2
al.constraint_tolerance_intermediate = 1e-1
al.kickout_max_penalty = false
al.verbose = true
opts = ALTROSolverOptions{Float64}(opts_al=al,
    projected_newton=false,
    projected_newton_tolerance=1e-2)
solver0 = solve!(prob, opts)
@test 1e-7 < max_violation(prob) < 1e-4
max_violation(prob)
solver0.stats[:time]



opts.projected_newton = true
opts.projected_newton_tolerance = 1e-2
opts.opts_pn.feasibility_tolerance = 1e-10
opts.opts_pn.active_set_tolerance = 1e-4
solver = SequentialNewtonSolver(prob, opts.opts_pn)
begin
    copyto!(solver.V.X, prob.X)
    copyto!(solver.V.U, prob.U)
    TO.newton_step!(prob, solver)
end
solver0 = ProjectedNewtonSolver(prob, opts.opts_pn)
TO.newton_step!(prob, solver)
TO.newton_step!(prob, solver0)

TO.update!(res, solver0)
Y,y = TO.active_constraints(res, solver0)
Hinv = inv(Diagonal(solver0.H))
S = Y*Hinv*Y'
cholesky(Symmetric(S))


res,solver = solve(prob, opts)
@test max_violation(res) < 1e-10z
solver.stats[:time]
@test solver.stats[:time] < solver0.stats[:time]


opts_pn = ProjectedNewtonSolverOptions{Float64}()
solver = SequentialNewtonSolver(res, opts_pn)
TO.newton_step!(res, solver)


opts = ALTROSolverOptions{Float64}(projected_newton=false, projected_newton_tolerance=1e-1, opts_al=opts_al)
res, = solve(prob, opts)
max_violation(res)

solver = SequentialNewtonSolver(res, opts.opts_pn)
solve(res, solver)
