
# Set up problem
prob = copy(Dynamics.quadrotor_obstacles)
n,m,N = size(prob)
add_constraints!(prob, bound_constraint(n, m, u_min=0, u_max=10))

# initial state
x0 = zeros(n)
x0[1:3] = [0.; 0.; 0.]
q0 = [1.;0.;0.;0.]
x0[4:7] = q0

# -final state
xf = copy(x0)
xf[1:3] = [0.;40.;0.] # xyz position
xf[4:7] = q0

# Create DIRCOL Solver
opts = DIRCOLSolverOptions{Float64}()
rollout!(prob)
dircol = DIRCOLSolver(prob, opts)
prob.constraints

N = prob.N
length(dircol.Z.U) == N
part_z = create_partition(n,m,N,N)
NN = N*(n+m)
# NN = N*n + (N-1)*m
Z = rand(NN)

# Convert Z to X,U
Z0 = Primals(Z,n,m)
Z0 = Primals(Z,part_z)
X = Z0.X
U = Z0.U

Z1 = copy(Z)
Z0 = Primals(Z1,X,U)

# Convert fom X,U to Z
X = [rand(n) for k = 1:N]
U = [rand(n) for k = 1:N]
Z = Primals(X,U)
Primals(prob, true).U
dircol.Z.U

# Test methods

# Cost
Z = dircol.Z
Z.X isa AbstractVectorTrajectory
typeof(Z.X) <: Vector{S} where S <: AbstractVector
cost(prob, dircol) == cost(prob)

# Collocation constraints
g_colloc = zeros(n*(N-1))
dynamics!(prob, dircol)
traj_points!(prob, dircol)
collocation_constraints!(g_colloc, prob, dircol)
g_colloc

eval_g(Z.Z,g)
g[1:length(g_colloc)] == g_colloc

# All Constraints


# Ipopt
method = :hermite_simpson
obj = Dynamics.quadrotor_3obs[2]
solver = Solver(prob.model,obj,N=N,integration=:rk3)
nG, = get_nG(solver,method)
U = ones(m,N)*1
X = line_trajectory(obj.x0, obj.xf, N)
X[4,:] .= 1
Z = packZ(X,U)

eval_f, eval_g, eval_grad_f, eval_jac_g = gen_usrfun_ipopt(solver,method)
x_L, x_U, g_L, g_U = get_bounds(solver,method)
P = length(g_L)  # Total number of constraints
N,N_ = get_N(solver,method)

g = zeros(P)
grad_f = zeros(NN)
rows = zeros(nG)
cols = zeros(nG)
vals = zeros(nG)
fVal = zeros(n,N)

eval_g(Z,g)
g


g_colloc = zeros((N-1)*n)
fVal = zeros(eltype(Z),n,N)
X_ = zeros(eltype(Z),n,N_)
U_ = zeros(eltype(Z),m,N_)
fVal_ = zeros(eltype(Z),n,N_)

X_,U_ = get_traj_points(solver,X,U,fVal,X_,U_,method)
get_traj_points_derivatives!(solver,X_,U_,fVal_,fVal, method)
collocation_constraints!(solver,X_,U_,fVal_,g_colloc,method)
g_colloc

N,N_ = get_N(prob, solver)
nG = TrajectoryOptimization.get_nG(prob, solver)

TrajectoryOptimization.solve_ipopt(prob, solver)

fVal = zeros(n,N)
gX_,gU_,fVal_ = TrajectoryOptimization.init_traj_points(prob,solver,fVal)
TrajectoryOptimization.init_jacobians(prob, solver)


include("../src/newton_nlp.jl")
X = rand(n,N)
U = rand(m,N)
Z0 = PrimalVars(X, U)
Z = packZ(X,U)
PrimalVars(Z, n, m, N)
@btime PrimalVars($Z, $n, $m, $N)

ix, iu = get_primal_inds(n, m, N, N-1)
z_part = (X=ix, U=iu, Z=1:NN)
Z1 = BlockArray(Z, z_part)
Z1.X == Z0.X
Z1.U == Z0.U
Z1.Z == Z
@btime BlockArray($Z, $z_part)
@btime $Z0.X
@btime $Z1.X
