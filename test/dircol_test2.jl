using ForwardDiff
using SparseArrays
using Test
using Ipopt
using LinearAlgebra
using PartedArrays
using Plots
import TrajectoryOptimization: Objective, update_problem, ProblemConstraints, Primals, unpackZ, dynamics
import TrajectoryOptimization: traj_points, collocation_constraints!, num_colloc

## Box parallel park
T = Float64
model = TrajectoryOptimization.Dynamics.car_model
n = model.n; m = model.m
model_d = Model{Discrete}(model,rk4)


# cost
x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
Qf = 100.0*Diagonal(I,n)
Q = (1e-2)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)
lqr_cost = LQRCost(Q,R,Qf,xf)


# constraints
u_bnd = 2.
x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]
bnd = BoundConstraint(n,m,x_min=x_min,x_max=x_max,u_min=-u_bnd,u_max=u_bnd,trim=true)

goal_con = goal_constraint(xf)
con = [bnd, goal_con]

# problem
N = 51
U = [ones(m) for k = 1:N-1]
dt = 0.06



# Re-create the problem with rolled out trajectory
U0 = ones(m,N-1)
prob = Problem(model_d,Objective(lqr_cost,N),U0,constraints=ProblemConstraints(N),dt=dt,x0=x0)
prob.constraints[N] += goal_con
rollout!(prob)
prob = update_problem(prob, model=model)

problem = TrajectoryOptimization.gen_ipopt_prob(prob)
solveProblem(problem)


# Extract out X,U
n,m,N = size(prob)
NN = (n+m)N
dt = prob.dt
Z = Primals(prob, true).Z
part_z = create_partition(n,m,N,N)
X,U = unpackZ(Z, part_z)

# Get the cost
cost(prob.obj, X, U)

# Check mipoints
fVal = dynamics(prob, X, U)
xs = [x[1] for x in X]
ys = [x[2] for x in X]
scatter(xs,ys)
Xm = traj_points(prob, X, U, fVal)
xm = [x[1] for x in Xm]
ym = [x[2] for x in Xm]
scatter!(xm,ym)

# Check collocation constraints
p_colloc = num_colloc(prob)
p_colloc == (N-1)*n
g = zeros(p_colloc)
TrajectoryOptimization.collocation_constraints!(g, prob, X, U)
x1,u1 = X[1], U[1];
x2,u2 = X[2], U[2];
f1,f2,fm = zero(x1), zero(x2), zero(x1);
evaluate!(f1, model, x1, u1)
evaluate!(f2, model, x2, u2)
f1 == fVal[1]
f2 == fVal[2]
xm = 0.5*(x1+x2) + dt/8*(f1-f2)
xm == Xm[1]
um = (u1+u2)/2
evaluate!(fm, model, xm, um)

err1 = x2-x1 - dt/6*(f1 + 4fm + f2)
@test norm(err1) <= 1e-15
@test norm(g) < 1e-15

# Cost gradient
function eval_cost(Z)
    X,U = unpackZ(Z,part_z)
    cost(prob.obj, X, U)
end
grad_f = zeros(NN)
TrajectoryOptimization.cost_gradient!(grad_f, prob, X, U)

@test ForwardDiff.gradient(eval_cost, Z) ≈ grad_f

# Collocation jacobian
function colloc(Z)
    X,U = unpackZ(Z,part_z)
    g = zeros(eltype(Z),p_colloc)
    collocation_constraints!(g, prob, X, U)
    return g
end
jac_co = zeros(p_colloc, NN)
TrajectoryOptimization.collocation_constraint_jacobian!(jac_co, prob, X, U)
@test ForwardDiff.jacobian(colloc, Z) ≈ jac_co

jac_struct = spzeros(p_colloc, NN)
TrajectoryOptimization.collocation_constraint_jacobian_sparsity!(jac_struct, prob)
TrajectoryOptimization.get_rc(jac_struct)

eval_f2, eval_g, eval_grad_f, eval_jac_g = TrajectoryOptimization.gen_ipopt_functions2(prob)

P = p_colloc
nG = p_colloc*2(n+m)
nH = 0
z_L = ones(NN)*-1e5
z_U = ones(NN)*1e5
g_L = zeros(P)
g_U = zeros(P)
z_L[1:n] = x0
z_U[1:n] = x0
problem = Ipopt.createProblem(NN, z_L, z_U, P, g_L, g_U, nG, nH,
    eval_f2, eval_g, eval_grad_f, eval_jac_g)
opt_file = joinpath(TrajectoryOptimization.root_dir(),"ipopt.opt")
addOption(problem,"option_file_name",opt_file)

Zinit = Primals([zeros(n) for k = 1:N], [ones(m) for k = 1:N])
problem.x = Zinit.Z

solveProblem(problem)
Zsol = Primals(problem.x, Zinit)
plot(Zsol.X)



# Manual Test
N = 11
n,m = 3,2
NN = (n+m)N
p_colloc = (N-1)*n
P = p_colloc
nG = p_colloc*2(n+m)
dt = 0.05

jac_struct = jac_g_structure(n,m,N)
r,c = TrajectoryOptimization.get_rc(jac_struct)

part_z = create_partition(n,m,N,N)
X0 = [zeros(n) for k = 1:N]
U0 = [ones(m) for k = 1:N]
Z0 = packZ(X0, U0, part_z)
X,U = unpackZ(Z, part_z)

function eval_f(Z)
    X,U = unpackZ(Z, part_z)
    N = length(X)

    J = 0.0
    for k = 1:N-1
        J += X[k]'Q*X[k] + U[k]'R*U[k]
    end
    J += (X[N] - xf)'Qf*(X[N] - xf)
    return J/2
end

function eval_g(Z::Vector{T},g) where T
    X,U = unpackZ(Z, part_z)
    N = length(X)
    n,m = length(X[1]), length(U[1])

    fVals = [zeros(T,n) for k = 1:N]
    for k = 1:N
        evaluate!(fVals[k], model, X[k], U[k])
    end
    g_ = reshape(g, n, N-1)
    for k = 1:N-1
        xm = 0.5*(X[k] + X[k+1]) + dt/8*(fVals[k] - fVals[k+1])
        um = 0.5*(U[k] + U[k+1])
        fValm = zero(X[k])
        evaluate!(fValm, model, xm, um)
        g_[:,k] = X[k] - X[k+1] + dt*(fVals[k] + 4fValm + fVals[k+1])
    end
end

eval_g2(g, Z) = eval_g(Z, g)

function eval_grad_f(Z, grad_f)
    X,U = unpackZ(Z, part_z)
    N = length(X)
    n,m = length(X[1]), length(U[1])

    grad = reshape(grad_f, n+m, N)
    for k = 1:N-1
        grad[:,k] = [Q*X[k]; R*U[k]]
    end
    grad[:,N] = [Qf*(X[N] - xf); zeros(m)]
end

function eval_jac_g(Z, mode, row, col, val)
    if mode == :Structure
        copyto!(row, r)
        copyto!(col, c)
    else
        jac = zeros(p_colloc, length(Z))
        g = zeros(p_colloc)
        ForwardDiff.jacobian!(jac, eval_g2, g, Z)
        copyto!(val, jac[CartesianIndex.(r,c)])
    end
end

function jac_g_structure(n,m,N)
    p_colloc = (N-1)*n
    NN = N*(n+m)
    jac = spzeros(p_colloc, NN)

    n_blk = 2(n+m)n
    blk = reshape(1:n_blk, n, 2(n+m))
    off1 = 0
    off2 = 0
    off = 0
    for k = 1:N-1
        jac[off1 .+ (1:n), off2 .+ (1:2(n+m))] = blk .+ off
        off1 += n
        off2 += n+m
        off  += n_blk
    end
    return jac
end

grad_f = zeros(NN)
eval_grad_f(Z, grad_f)
ForwardDiff.gradient(eval_f, Z) ≈ grad_f

g = zeros(P)
eval_g(Z,g)

row,col,val = zeros(nG), zeros(nG), zeros(nG)
eval_jac_g(Z, :Structure, row, col, val)
eval_jac_g(Z, :Vals, row, col, val)
r
c
