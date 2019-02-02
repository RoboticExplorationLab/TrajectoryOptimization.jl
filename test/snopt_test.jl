using Snopt
using SparseArrays

model, obj = Dynamics.dubinscar_parallelpark

solver = Solver(model, obj, N=101)
n,m,N = get_sizes(solver)
X0 = line_trajectory(solver)
U0 = ones(m,N-1)

solver.opts.cost_tolerance = 1e-6
solver.opts.cost_tolerance_intermediate = 1e-4
solver.opts.constraint_tolerance = 1e-4
solver.opts.square_root = true
solver.opts.penalty_initial = 0.1
solver.opts.penalty_scaling = 20
res, stats = solve(solver,U0)
stats["runtime"]
_cost(solver, res)

method = :hermite_simpson
N,N_ = get_N(solver,method)
n,m = get_sizes(solver)

# Create results structure
results = DircolResults(n,m,solver.N,method)
var0 = DircolVars(X0,U0)
Z0 = var0.Z

# Generate the objective/constraint function and its gradients
usrfun = gen_usrfun(solver, results, method, grads=:quad)
J, c, ceq, grad_J, jacob_c, jacob_ceq, fail = usrfun(Z0)

NN = (n+m)N
nG, = get_nG(solver,method)
eval_f, eval_g, eval_grad_f, eval_jac_g = gen_usrfun_ipopt(solver,method)
x_L, x_U, g_L, g_U = get_bounds(solver,method)
P = length(g_L)  # Total number of constraints
g = zeros(P)
grad_f = zeros(NN)
rows = zeros(nG)
cols = zeros(nG)
vals = zeros(nG)

eval_f(Z0) ≈ J
eval_g(Z0,g)
g == ceq
eval_grad_f(Z0, grad_f)
grad_f == grad_J
eval_jac_g(Z0,:Structure,rows,cols,vals)
eval_jac_g(Z0,:vals,rows,cols,vals)
cJ = sparse(rows,cols,vals)
jacob_ceq == cJ

function eval_all(Z0)
    eval_f(Z0)
    eval_g(Z0,g)
    eval_grad_f(Z0,grad_f)
    eval_jac_g(Z0,:vals,rows,cols,vals)
end

@btime usrfun($Z0)
@btime eval_all($Z0)



# Set up the problem
lb,ub = get_bounds(solver,method)

options = Dict{String, Any}()
options["Derivative option"] = 0
options["Verify level"] = 1
options["Minor feasibility tol"] = solver.opts.constraint_tolerance
# options["Minor optimality  tol"] = solver.opts.eps_intermediate
options["Major optimality  tol"] = solver.opts.cost_tolerance



row,col = constraint_jacobian_sparsity(solver,method)
# prob = Snopt.createProblem(usrfun, Z0, lb, ub, iE=row, jE=col)
# prob = Snopt.createProblem(usrfun, Z0, lb, ub)
# prob.x = Z0
# t_eval = @elapsed z_opt, fopt, info = snopt(prob, options, start=start)

usrfun = gen_usrfun(solver, results, method, grads=:quad)
@time z_opt, fopt, info = snopt(usrfun, Z0, lb, ub, options)
var_opt = DircolVars(z_opt,n,m,N)


cost(solver, var_opt)
_cost(solver, res)

res_s, stats_s = solve_dircol(solver, X0, U0, nlp=:snopt)
stats_s["runtime"]
res_p, stats_p = solve_dircol(solver, X0, U0, nlp=:ipopt)
stats_p["runtime"]
stats["runtime"]
parse_snopt_summary()

p = plot()
plot_trajectory!(res_s.X,label="snopt")
plot_trajectory!(res_p.X,label="ipopt")
plot_trajectory!(res,label="iLQR")
