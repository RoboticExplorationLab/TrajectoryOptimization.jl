import TrajectoryOptimization: DIRCOLSolver, DIRCOLSolverOptions

# Set up problem
prob = Dynamics.quadrotor_obstacles
n,m,N = size(prob)
add_constraints!(prob, bound_constraint(n, m, u_min=0, u_max=10))

# Create DIRCOL Solver
opts = DIRCOLSolverOptions{Float64}()
rollout!(prob)
solver = DIRCOLSolver(prob, opts)

N = prob.N
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
Primals(prob)

# Test methods
Z = solver.Z
Z.X isa AbstractVectorTrajectory
typeof(Z.X) <: Vector{S} where S <: AbstractVector
cost(prob, solver) == cost(prob)


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
