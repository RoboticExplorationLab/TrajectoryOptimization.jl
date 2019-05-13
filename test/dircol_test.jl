
# Set up problem
prob = copy(Dynamics.quadrotor_obstacles)
model = Dynamics.quadrotor_model
n,m,N = size(prob)
bnd = bound_constraint(n, m, u_min=0, u_max=10)
add_constraints!(prob, bnd)


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
prob = update_problem(prob, model=model)
pcon = prob.constraints
dircol = DIRCOLSolver(prob, opts)

N = prob.N
length(dircol.Z.U) == N
part_z = create_partition(n,m,N,N)
NN = N*(n+m)
Z = rand(NN)
copyto!(dircol.Z.Z, Z)

# Convert Z to X,U
Z0 = Primals(Z,n,m)
Z0 = Primals(Z,part_z)
X = Z0.X
U = Z0.U

Primals(Z,n,m).X[1][1] = 10
@test X[1][1] == 10
@test Z[1] == 10

Z1 = copy(Z)
Z1[1] = 100
Z1 = Primals(Z1,X,U)
@test Z1.Z[1] == 100
@test Z1.X[1][1] == 100
@test X[1][1] == 100


# Convert fom X,U to Z
X = [rand(n) for k = 1:N]
U = [rand(n) for k = 1:N]
Z2 = Primals(X,U)
@test Z2.equal == true
@test Primals(prob, true).equal == true
Z_rollout = Primals(prob, true)


# Test methods

# Cost
Z = dircol.Z
Z.X isa AbstractVectorTrajectory
typeof(Z.X) <: Vector{S} where S <: AbstractVector
initial_controls!(prob, Z.U)
initial_state!(prob, Z.X)
@test cost(prob, dircol) == cost(prob)

# Collocation constraints
p_colloc = n*(N-1)
p_custom = sum(num_constraints(prob))
g_colloc = zeros(p_colloc)
g = zeros(p_colloc + p_custom)
dynamics!(prob, dircol, Z_rollout)
traj_points!(prob, dircol, Z_rollout)
collocation_constraints!(g_colloc, prob, dircol, Z_rollout)
@test norm(g_colloc) == 0

# Normal Constraints
dynamics!(prob, dircol, Z)
traj_points!(prob, dircol, Z)
g_custom = view(g, p_colloc.+(1:p_custom))
update_constraints!(g, prob, dircol)
update_constraints!(dircol.C, prob.constraints, Z.X, Z.U)
p = num_constraints(prob)
@test to_dvecs(g_custom, p) == dircol.C

# Cost gradient
grad_f = zeros(NN)
cost_gradient!(grad_f, prob, dircol)

function eval_f(Z)
    Z = Primals(Z, part_z)
    cost(prob.obj, Z.X, Z.U)
end
eval_f(Z.Z)
@test ForwardDiff.gradient(eval_f, Z.Z) == grad_f

# Collocation Constraint jacobian
dircol = DIRCOLSolver(prob, opts, Z)
dynamics!(prob, dircol, Z)
traj_points!(prob, dircol, Z)
calculate_jacobians!(prob, dircol, Z)

jac = zeros(p_colloc, NN)
collocation_constraint_jacobian!(jac, prob, dircol)

function jac_colloc(Z)
    dynamics!(prob, dircol)
    traj_points!(prob, dircol)
    calculate_jacobians!(prob, dircol, Z)

    jac = zeros(p_colloc, NN)
    collocation_constraints!(g_colloc, prob, dircol, Z)
    collocation_constraint_jacobian!(jac, prob, dircol)
    return jac
end
jac_colloc(Z)

function c_colloc(Z)
    g_colloc = zeros(eltype(Z), p_colloc)
    Z = Primals(Z, part_z)
    solver = DIRCOLSolver(prob, DIRCOLSolverOptions{eltype(Z.Z)}(), Z)
    dynamics!(prob, solver, Z)
    traj_points!(prob, solver, Z)
    collocation_constraints!(g_colloc, prob, solver, Z)
    return g_colloc
end
c_colloc(Z.Z)

@test ForwardDiff.jacobian(c_colloc, Z.Z) ≈ jac_colloc(Z)

# General constraint jacobian
jac = zeros(p_colloc+p_custom, NN)
calculate_jacobians!(prob, dircol, Z)
constraint_jacobian!(jac, prob, dircol, Z)

function c_all(Z)
    g = zeros(eltype(Z), p_colloc+p_custom)
    Z = Primals(Z, part_z)
    solver = DIRCOLSolver(prob, DIRCOLSolverOptions{eltype(Z.Z)}(), Z)
    dynamics!(prob, solver, Z)
    traj_points!(prob, solver, Z)
    update_constraints!(g, prob, solver, Z)
    return g
end
c_all(Z.Z)
@test ForwardDiff.jacobian(c_all, Z.Z) ≈ jac
