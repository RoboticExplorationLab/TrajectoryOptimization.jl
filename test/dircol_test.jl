import TrajectoryOptimization: get_dt_traj

# Set up problem
model = Dynamics.car
n,m = model.n, model.m
Q = (1e-2)*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)
x0 = [0.;0.;0.]
xf = [0.;1.;0.]

xf = [0,1.0,0]
goal_con = goal_constraint(xf)
circle_con = TO.planar_obstacle_constraint(model.n, model.m, (0,2.5), 0.25)
bnd = BoundConstraint(model.n, model.m, x_min=[-0.5,-0.001,-Inf], x_max=[0.5, 1.001, Inf], u_min=-2, u_max=2)

# Initial Controls
N = 101
U0 = [ones(model.m) for k = 1:N]
obj = TrajectoryOptimization.LQRObjective(Q,R,Qf,xf,N)


# Create Problem
n,m = model.n, model.m
prob = Problem(rk3(model), obj, constraints=Constraints([bnd,circle_con],N), N=N, tf=3.0)
prob.constraints[N] += goal_con
initial_controls!(prob, U0)
rollout!(prob)
prob = TO.update_problem(prob, model=model)


# Create DIRCOL Solver
opts = TO.DIRCOLSolverOptions{Float64}(verbose=false)
pcon = prob.constraints
dircol = TO.DIRCOLSolver(prob, opts)

N = prob.N
@test length(dircol.Z.U) == N
part_z = create_partition(n,m,N,N)
NN = N*(n+m)
Z = rand(NN)

# Convert Z to X,U
Z0 = Primals(Z,n,m)
Z0 = Primals(Z,part_z)
X = Z0.X
U = Z0.U
@test size(Z0) == (n,m,N)
@test length(Z0) == NN

Primals(Z,n,m).X[1][1] = 10
@test X[1][1] == 10
@test Z[1] == 10

Z1 = copy(Z)
Z1[1] = 100
Z1 = Primals(Z1,X,U)
@test Z1.Z[1] == 100
@test Z1.X[1][1] == 100
@test X[1][1] == 100

Z2 = Primals(Z, Z0)
@test Z2.Z == Z

# Convert fom X,U to Z
X = [rand(n) for k = 1:N]
U = [rand(m) for k = 1:N]
Z2 = Primals(X,U)
@test Z2.equal == true
@test Primals(prob, true).equal == true
Z_rollout = Primals(prob, true)

Z3 = TO.pack(prob)
@test length(Z3) == NN

@test TO.pack(X,U, part_z) == Z2.Z
@test TO.pack(X,U) == Z2.Z




# Test methods

# Cost
Z = Primals(prob, true)
X,U = Z.X, Z.U
@test Z.X isa TrajectoryOptimization.AbstractVectorTrajectory
@test typeof(Z.X) <: Vector{S} where S <: AbstractVector
initial_controls!(prob, Z.U)
TO.initial_states!(prob, Z.X)
@test cost(prob, dircol) == cost(prob)

# Collocation constraints
p_colloc = n*(N-1)
p_custom = sum(num_constraints(prob))
g_colloc = zeros(p_colloc)
g = zeros(p_colloc + p_custom)
# TO.dynamics!(prob, dircol, X, U)
# traj_points!(prob, dircol, Z_rollout)
TO.collocation_constraints!(g_colloc, prob, dircol, X, U)
@test norm(g_colloc,Inf) < 1e-15

# Normal Constraints
g_custom = view(g, p_colloc.+(1:p_custom))
TO.update_constraints!(g, prob, dircol, X, U)
p = num_constraints(prob)
@test p == dircol.p

# Cost gradient
# grad_f = zeros(NN)
# TO.cost_gradient!(grad_f, prob, X, U, get_dt_traj(prob))

# Z = Primals(Z, part_z)
#
# cost(prob.obj, Z.X, Z.U, get_dt_traj(prob))
#
function eval_f(Z)
    Z = Primals(Z, part_z)
    cost(prob.obj, Z.X, Z.U, get_dt_traj(prob))
end
eval_f(Z.Z)
# @test ForwardDiff.gradient(eval_f, Z.Z) == grad_f

# Collocation Constraint jacobian
nG_colloc = p_colloc*2(n+m)
dircol = DIRCOLSolver(prob, opts, Z)

jac = zeros(p_colloc, NN)
TO.collocation_constraint_jacobian!(jac, prob, dircol, X, U)
jac

function jac_colloc(Z)
    X,U = TO.unpack(Z, part_z)
    rows,cols = zeros(nG_colloc), zeros(nG_colloc)
    jac_struct = spzeros(p_colloc,NN)
    TO.collocation_constraint_jacobian_sparsity!(jac_struct, prob)
    r,c = TO.get_rc(jac_struct)
    jac = zeros(nG_colloc)
    TO.collocation_constraint_jacobian!(jac, prob, dircol, X, U)
    return sparse(r,c,jac)
end
jac_colloc(Z.Z)


function c_colloc(Z)
    X,U = TO.unpack(Z, part_z)
    g_colloc = zeros(eltype(Z), p_colloc)
    Z = Primals(Z, part_z)
    solver = DIRCOLSolver(prob, DIRCOLSolverOptions{eltype(Z.Z)}(), Z)
    TO.collocation_constraints!(g_colloc, prob, solver, X, U)
    return g_colloc
end
c_colloc(Z.Z)

@test ForwardDiff.jacobian(c_colloc, Z.Z) ≈ Array(jac_colloc(Z.Z))


# General constraint jacobian
nG_custom = sum(p[1:N-1])*(n+m) + p[N]*n
nG = nG_colloc + nG_custom
jac = spzeros(p_colloc+p_custom, NN)
TO.constraint_jacobian_sparsity!(jac, prob)
rows,cols = TO.get_rc(jac)

function jac_con(Z)
    X,U = TO.unpack(Z, part_z)

    jac = zeros(nG)
    nG_colloc = p_colloc * 2(n+m)
    jac_colloc = view(jac, 1:nG_colloc)
    solver = DIRCOLSolver(prob, DIRCOLSolverOptions{eltype(Z)}(), Primals(Z, part_z))
    TO.collocation_constraint_jacobian!(jac_colloc, prob, solver, X, U)

    # General constraint jacobians
    jac_custom = view(jac, nG_colloc+1:length(jac))
    TO.constraint_jacobian!(jac_custom, prob, solver, X, U)

    return sparse(rows,cols,jac)
end
jac_con(Z.Z)

function c_all(Z)
    X,U = TO.unpack(Z, part_z)
    g = zeros(eltype(Z), p_colloc+p_custom)
    g_colloc = view(g,1:p_colloc)
    g_custom = view(g,p_colloc+1:length(g))
    solver = DIRCOLSolver(prob, DIRCOLSolverOptions{eltype(Z)}(), Primals(Z, part_z))
    TO.collocation_constraints!(g_colloc, prob, solver, X, U)
    TO.update_constraints!(g_custom, prob, solver, X, U)
    return g
end
g = c_all(Z.Z)
@test ForwardDiff.jacobian(c_all, Z.Z) ≈ jac_con(Z.Z)

# Test Ipopt functions
eval_f2, eval_g, eval_grad_f, eval_jac_g = TO.gen_dircol_functions(prob, dircol)

g2 = zero(g)
# grad_f2 = zero(grad_f)
jac2 = zeros(nG)
row, col = zeros(nG), zeros(nG)
@test eval_f2(Z.Z) == eval_f(Z.Z)
eval_g(Z.Z, g2)
@test g ≈ g2
# eval_grad_f(Z.Z, grad_f2)
# @test grad_f ≈ grad_f2

eval_jac_g(Z.Z, :Structure, row, col, jac2)
eval_jac_g(Z.Z, :Values, row, col, jac2)
@test sparse(row,col,jac2) == jac_con(Z.Z)


# Strip out state and control Bounds
prob0 = copy(prob)
p0 = num_constraints(prob0)
bnds = TO.remove_bounds!(prob0)
p = num_constraints(prob0)
@test p0[1] == 9
@test p[1] == 1
@test p[N] == 1
z_U,z_L,g_U,g_L = TO.get_bounds(prob0,bnds)
Z_U = Primals(z_U, part_z)
Z_L = Primals(z_L, part_z)
@test Z_U.X[1] == Z_L.X[1] == prob0.x0
@test Z_U.X[N] == Z_L.X[N] == xf
@test Z_U.U[N] == Z_U.U[N-1]

# @test_nowarn solve!(prob, opts)
# @test_nowarn TO.write_ipopt_options()
