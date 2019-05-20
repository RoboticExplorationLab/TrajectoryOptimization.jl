using TrajectoryOptimization
const TO = TrajectoryOptimization
using ForwardDiff
using SparseArrays
using Ipopt
using Plots
using LinearAlgebra
using PartedArrays

# Set up problem
model = Dynamics.car_model
costfun = Dynamics.car_costfun
xf = [0,1.0,0]
goal_con = goal_constraint(xf)
circle_con = TO.planar_obstacle_constraint(model.n, model.m, (0,0.5), 0.25)
bnd = BoundConstraint(model.n, model.m, x_min=[-0.5,-0.001,-Inf], x_max=[0.5, 1.001, Inf], u_min=-2, u_max=2)

N = 51
n,m = model.n, model.m
prob = Problem(rk3(model), Objective(costfun,N), constraints=ProblemConstraints([bnd,circle_con],N), N=N, tf=3.0)
prob.constraints[N] += goal_con
prob = TO.update_problem(prob, model=model)


# Initial Guess
X0 = [zeros(model.n) for k = 1:N]
X0 = line_trajectory(prob.x0, xf, N)
U0 = [ones(model.m) for k = 1:N]
Z0 = TO.pack(X0,U0)

# Solve with Ipopt
sol, problem = TO.solve_ipopt(prob)
plot(sol.U)
plot()
TO.plot_circle!((0,0.5),0.25)
plot_trajectory!(sol.X)

# Strip out state and control Bounds
prob = copy(prob)
bnds = TO.remove_bounds!(prob)
part_z = create_partition(n,m,N,N)

# Generate Ipopt functions
eval_f, eval_g, eval_grad_f, eval_jac_g = TO.gen_ipopt_functions3(prob)

# Initialize Arrays
p = num_constraints(prob)
pcum = [0; cumsum(p)]
p_colloc = TO.num_colloc(prob)
p_custom = sum(num_constraints(prob))
grad_f = zeros(length(Z0))
g = zeros(TO.num_colloc(prob) + sum(p))
nG = TO.num_colloc(prob)*2*(model.n + model.m) + sum(p[1:N-1])*(n+m) + p[N]*n
rows, cols, vals = zeros(nG), zeros(nG), zeros(nG)

# Test functions
eval_f(Z0)
eval_grad_f(Z0, grad_f)
eval_g(Z0, g)
eval_jac_g(Z0, :Structure, rows, cols, vals)
eval_jac_g(Z0, :Values, rows, cols, vals)

@code_warntype eval_f(Z0)
@code_warntype eval_g(Z0,g)
@code_warntype eval_jac_g(Z0, :Structure, rows, cols, vals)
@code_warntype eval_jac_g(Z0, :Values, rows, cols, vals)
@code_warntype eval_grad_f(Z0, grad_f)

# General New Ipopt function
solver = TO.DIRCOLSolver(prob)
eval_f3, eval_g3, eval_grad_f3, eval_jac_g3 = TO.gen_ipopt_functions3(prob, solver)

# Compare output
g3 = zero(g)
grad_f3 = zero(grad_f)
rows3,cols3,vals3 = zero(rows), zero(cols), zero(vals)
eval_f(Z0) ≈ eval_f3(Z0)
eval_grad_f(Z0, grad_f); eval_grad_f3(Z0, grad_f3)
grad_f ≈ grad_f3
eval_g(Z0, g); eval_g3(Z0, g3);
g ≈ g3
eval_jac_g(Z0, :Structure, rows, cols, vals); eval_jac_g3(Z0, :Structure, rows3, cols3, vals3);
eval_jac_g(Z0, :Vals, rows, cols, vals); eval_jac_g3(Z0, :Vals, rows3, cols3, vals3);
sparse(rows,cols,vals) ≈ sparse(rows3,cols3,vals3)


# Validate derivatives
ForwardDiff.gradient(eval_f, Z0) ≈ grad_f

jac = zeros(length(g),length(Z0))
eval_g2(g,Z0) = eval_g(Z0,g)
eval_g2(g,Z0)
g
ForwardDiff.jacobian!(jac, eval_g2, g, Z0) ≈ Array(sparse(rows,cols,vals))

problem = TO.gen_ipopt_prob3(prob, xf)
problem.x = copy(Z0)
solveProblem(problem)
Zsol = TO.Primals(problem.x, model.n, model.m)

prob_d = Problem(rk3(model), Objective(costfun,N), constraints=ProblemConstraints([bnd,circle_con],N), N=N, tf=3.0)
initial_controls!(prob_d, U0)
ilqr = iLQRSolverOptions()
al = AugmentedLagrangianSolverOptions{Float64}()
AL = AugmentedLagrangianSolver(prob_d, al)
res = solve(prob_d, AL)
cost(res)
norm(res.X[end] - xf)
plot_trajectory!(res.X,markershape=:circle)
max_violation(res)


res2 = copy(res)
copyto!(res2.U, Zsol.U[1:N-1])
copyto!(res2.X, Zsol.X)
cost(res2)
TO.projection!(res2, ilqr)
cost(res2)
plot_trajectory!(res2.X)
