
prob = copy(Problems.cartpole)
prob = update_problem(prob, model=Dynamics.cartpole)
bnds = remove_bounds!(prob)
z_U, z_L, g_U, g_L = get_bounds(prob, bnds)


sprob = copy(Problems.cartpole_static)
n,m,N = size(sprob)
hs_con = ConstraintVals( ExplicitDynamics{HermiteSimpson}(sprob.model, sprob.N), 1:sprob.N)
add_constraint!(get_constraints(sprob), hs_con, 1)

xlim = @SVector [10,Inf,Inf,Inf]
ulim = @SVector [2]
bnd1 = StaticBoundConstraint(n,m, u_min=-ulim/2, u_max=ulim/2)
bnd2 = StaticBoundConstraint(n,m, x_min=-xlim, x_max=xlim, u_min=-ulim, u_max=ulim)
bnd3 = StaticBoundConstraint(n,m, x_min=-xlim, x_max=xlim)
con1 = ConstraintVals(bnd1,1:50)
con2 = ConstraintVals(bnd2,70:101)
con3 = ConstraintVals(bnd3,45:70)
con4 = ConstraintVals(GoalConstraint(sprob.xf), N:N)

# Test bounds
conSet = ConstraintSets([con1,con2,con3,con4],N)
@test_throws AssertionError get_bounds(conSet)

conSet = ConstraintSets([con1,con2,con4],N)
zU,zL,gU,gL = get_bounds(conSet)
@test (gU,gL) == get_bounds(conSet)[3:4]
@test gU == zero(gU)


# Test Copying from vector to trajectory
sprob = copy(Problems.cartpole_static)
n,m,N = size(sprob)
Z = sprob.Z
P = StaticPrimals(n,m,N)
NN = (n+m)*N
V = rand(NN)

x = state(Z[1])
u = control(Z[1])

copyto!(Z, V, P.xinds, P.uinds)
@test Z[1].z[1:n] == V[1:n]
@test Z[N].z == V[end-n-m+1:end]

@test (@allocated copyto!(Z, V, P.xinds, P.uinds)) == 0


# Compare Solves
prob = copy(Problems.cartpole)
prob = update_problem(prob,model=Dynamics.cartpole)
sprob = copy(Problems.cartpole_static)
rollout!(sprob)
initial_states!(prob, state(sprob))
initial_controls!(prob, control(sprob))
state(sprob) ≈ prob.X
control(sprob) ≈ prob.U

# Add dynamics constraint
conSet = get_constraints(sprob)
hs_con = ConstraintVals( ExplicitDynamics{HermiteSimpson}(sprob.model, sprob.N), 1:sprob.N-1)
init_con = ConstraintVals( GoalConstraint(sprob.x0), 1:1)
add_constraint!(get_constraints(sprob), hs_con, 1)
add_constraint!(get_constraints(sprob), init_con, 1)

# Build NLP problem
bnds = remove_bounds!(prob)
dircol = DIRCOLSolver(prob)
z_U, z_L, g_U, g_L = get_bounds(prob, bnds)
d = DIRCOLProblem(prob, dircol, z_L, z_U, g_L, g_U)

ds = StaticDIRCOLProblem(sprob)

NN = num_primals(ds.solver)
NP = num_duals(ds.solver)

# Test bounds
d.zL == ds.zL
d.zU == ds.zU
d.gL == ds.gL
d.gU == ds.gU

# Test jacobian structure
d.jac_struct == ds.jac_struct

# Test cost
V = rand(NN)
X,U = unpack(V, d.part_z)
dt_traj = get_dt_traj(prob)
MOI.eval_objective(d, V) ≈ MOI.eval_objective(ds, V)

# Test cost gradient
d = DIRCOLProblem(prob, dircol, z_L, z_U, g_L, g_U)
grad_f = zeros(NN)
grad_fs = zeros(NN)
MOI.eval_objective_gradient(d, grad_f, V)
MOI.eval_objective_gradient(ds, grad_fs, V)
grad_f ≈ grad_fs

@btime MOI.eval_objective_gradient($d,  $grad_f,  $V)
@btime MOI.eval_objective_gradient($ds, $grad_fs, $V)  # 78x faster

# Test constraints
g = zeros(NP)
gs = zero(g)
MOI.eval_constraint(d, g, V)
MOI.eval_constraint(ds, gs, V)
gs ≈ g
@btime MOI.eval_constraint($d,  $g,  $V)
@btime MOI.eval_constraint($ds, $gs, $V) # 35x faster

# Test constraint jacobian
nG = length(d.jac_struct)
jac_g = zeros(nG)
jac_gs = zero(jac_g)
MOI.eval_constraint_jacobian(d, jac_g, V)
MOI.eval_constraint_jacobian(ds, jac_gs, V)
jac_g ≈ jac_gs

@btime MOI.eval_constraint_jacobian($d,  $jac_g,  $V)
@btime MOI.eval_constraint_jacobian($ds, $jac_gs, $V)  # 40x faster

model = build_moi_problem(d)
MOI.optimize!(model)

model2 = build_moi_problem(ds)
MOI.optimize!(model2)
@btime MOI.optimize!($model)
b = @benchmark MOI.optimize!($model2) # 25x faster, 1000x less memory
