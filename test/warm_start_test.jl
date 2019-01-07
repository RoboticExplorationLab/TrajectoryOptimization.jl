

N = 51
model, obj = Dynamics.dubinscar
obj_con = ConstrainedObjective(obj, u_min=-10, u_max=10)
solver = Solver(model,obj_con,N=N)
solver.opts.verbose = true
n,m = get_sizes(solver)

U0 = rand(m,N)
res, stats = solve(solver, U0)
res.μ[1]

solver_warm = Solver(model,obj_con,N=N)
solver_warm.opts.verbose = true
solver_warm.opts.penalty_initial = res.μ[1][1]
λ = deepcopy(res.λ)
push!(λ, res.λN)
res_warm, stats_warm = _solve(solver_warm, to_array(res.U))
res_warm, stats_warm = _solve(solver_warm, to_array(res.U),λ=λ)


to_array(res.X) == to_array(res_warm.X)
to_array(res.U) == to_array(res_warm.U)
to_array(res.C) == to_array(res_warm.C)
to_array(res.λ) == to_array(res_warm.λ)
(res.CN) == (res_warm.CN)
(res.λN) == (res_warm.λN)

max_violation(res_warm)
max_violation(res)

to_array(res.λ)
p = plot()
plot_trajectory!(res)
cost(solver, res)



U0_warm = to_array(res.U)
res_warm = init_results(solver, X0, U0_warm, λ=λ)
rollout!(res_warm,solver)
cost(solver,res_warm)


to_array(res.μ)
to_array(res_warm.μ)



U = to_array(res.U)
solve(solver, U)
res.λ
