using TrajectoryOptimization
const TO = TrajectoryOptimization
using ForwardDiff
using SparseArrays
using Ipopt
using Plots

# Set up problem
model = Dynamics.car_model
costfun = Dynamics.car_costfun
xf = [0,1,0]
goal_con = goal_constraint(xf)

N = 21
prob = Problem(rk3(model), Objective(costfun,N), constraints=ProblemConstraints(N), N=N, tf=3.0)
# prob.constraints[N] += goal_con
prob = TO.update_problem(prob, model=model)

# Initial Guess
X0 = [zeros(model.n) for k = 1:N]
U0 = [ones(model.m) for k = 1:N]
Z0 = TO.pack(X0,U0)

# Generate Ipopt functions
eval_f, eval_g, eval_grad_f, eval_jac_g = TO.gen_ipopt_functions3(prob)

# Initialize Arrays
grad_f = zeros(length(Z0))
g = zeros(TO.num_colloc(prob))
nG = TO.num_colloc(prob)*2*(model.n + model.m)
rows, cols, vals = zeros(nG), zeros(nG), zeros(nG)

# Test functions
eval_f(Z0)
eval_grad_f(Z0, grad_f)
eval_g(Z0,g)
eval_jac_g(Z0, :Structure, rows, cols, vals)
eval_jac_g(Z0, :Values, rows, cols, vals)

@code_warntype eval_f(Z0)
@code_warntype eval_grad_f(Z0, grad_f)
@code_warntype eval_g(Z0,g)
@code_warntype eval_jac_g(Z0, :Structure, rows, cols, vals)
@code_warntype eval_jac_g(Z0, :Values, rows, cols, vals)

# Validate derivatives
ForwardDiff.gradient(eval_f, Z0) ≈ grad_f

jac = zeros(length(g),length(Z0))
eval_g2(g,Z0) = eval_g(Z0,g)
ForwardDiff.jacobian!(jac, eval_g2, g, Z0) ≈ Array(sparse(rows,cols,vals))

n,m = model.n, model.m
NN = (n+m)N
P = TO.num_colloc(prob)
z_U = ones(NN)*Inf
z_L = ones(NN)*-Inf
g_U = zeros(P)
g_L = zeros(P)
z_U[1:n] = zeros(n)
z_L[1:n] = zeros(n)
z_U[(N-1)*(n+m) .+ (1:n)] = xf
z_L[(N-1)*(n+m) .+ (1:n)] = xf
problem = createProblem(NN, z_L, z_U, P, g_L, g_U, P*2(n+m), 0,
    eval_f, eval_g, eval_grad_f, eval_jac_g)
opt_file = joinpath(TrajectoryOptimization.root_dir(),"ipopt.opt");
addOption(problem,"option_file_name",opt_file)
problem.x = copy(Z0)

solveProblem(problem)
Zsol = TO.Primals(problem.x, n, m)
plot(Zsol.X)
plot()
plot_trajectory!(Zsol.X)

prob.constraints[N]
problem = TO.gen_ipopt_prob3(prob, xf)
problem.x = copy(Z0)
solveProblem(problem)
