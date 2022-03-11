# Model and discretization
model = Cartpole()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
n,m = RD.dims(model)
N = 11
tf = 5.
dt = tf/(N-1)

# Create vector of models
dmodels = [copy(dmodel) for k = 1:N-1]

# Initial and Final conditions
x0 = @SVector zeros(n)
xf = @SVector [0, pi, 0, 0]

# Objective
Q = 1.0e-2*Diagonal(@SVector ones(n))
Qf = 100.0*Diagonal(@SVector ones(n))
R = 1.0e-1*Diagonal(@SVector ones(m))
obj = LQRObjective(Q,R,Qf,xf,N)

# Constraints
u_bnd = 3.0
conSet = ConstraintList(n,m,N)
bnd = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
goal = GoalConstraint(xf)
add_constraint!(conSet, bnd, 1:N-1)
add_constraint!(conSet, goal, N:N)

# Initial conditions
X0 = [@SVector fill(0.0,n) for k = 1:N]
u0 = @SVector fill(0.01,m)
U0 = [u0 for k = 1:N-1]
Z = SampledTrajectory(X0,U0, dt=dt) 

# Inner constructor
prob = Problem(dmodels, obj, conSet, x0, xf, Z, N, 0.0, tf)
@test prob.x0 == x0
@test prob.xf == xf
@test prob.constraints === conSet
add_constraint!(conSet, goal, N-1)
@test length(TO.get_constraints(prob)) == length(conSet)
@test TO.num_constraints(TO.get_constraints(prob)) === conSet.p
@test prob.obj === obj
@test prob.tf ≈ tf
@test prob.N == N
@test states(prob) ≈ X0
@test controls(prob) ≈ U0

# Try passing a single model
prob = Problem(dmodel, obj, conSet, x0, xf, Z, N, 0.0, tf)
@test prob.x0 == x0
@test prob.xf == xf
@test prob.constraints === conSet
add_constraint!(conSet, goal, N-1)
@test length(TO.get_constraints(prob)) == length(conSet)
@test TO.num_constraints(TO.get_constraints(prob)) === conSet.p
@test prob.obj === obj
@test prob.tf ≈ tf
@test prob.N == N
@test states(prob) ≈ X0
@test controls(prob) ≈ U0

# Alternate constructor
prob = Problem(dmodel, obj, x0, tf, xf=xf, constraints=conSet, X0=X0, U0=U0)
@test prob.x0 == x0
@test prob.xf == xf
@test prob.constraints === conSet
add_constraint!(conSet, goal, N-1)
@test length(TO.get_constraints(prob)) == length(conSet)
@test TO.num_constraints(TO.get_constraints(prob)) === conSet.p
@test prob.obj === obj
@test prob.tf ≈ tf
@test prob.N == N
@test states(prob) ≈ X0
@test controls(prob) ≈ U0

# Change integration
prob = Problem(model, obj, x0, tf, xf=xf, constraints=conSet, integration=RD.Euler(model))
@test RD.integration(prob.model[1]) isa RD.Euler

# Test defaults
prob = Problem(model, obj, x0, tf)
@test prob.x0 == zero(x0)
@test prob.N == N
@test all(all.(isnan, states(prob)))
@test controls(prob) == [zeros(m) for k = 1:N-1]
@test isempty(prob.constraints)
@test RobotDynamics.gettimes(prob) ≈ range(0, tf; step=dt)

# Set initial trajectories
initial_states!(prob, 2 .* X0)
@test states(prob) ≈ 2 .* X0
initial_controls!(prob, 2 .* U0)
@test controls(prob) ≈ 2 .* U0

initial_trajectory!(prob, Z)
@test controls(prob) ≈ U0
@test states(prob) ≈ X0

# Use 2D matrices
X0_mat = hcat(X0...)
U0_mat = hcat(U0...)
initial_states!(prob, 3*X0)
initial_controls!(prob, 4*U0)
@test states(prob) ≈ 3 .* X0
@test controls(prob) ≈ 4 .* U0

prob = Problem(model, obj, xf, tf, X0=X0_mat, U0=U0_mat)
@test states(prob) ≈ X0
@test controls(prob) ≈ U0

# Use variable time steps
dts = rand(N-1)
dts = dts / sum(dts) * tf
prob = Problem(model, obj, xf, tf, dt=dts)
times = RobotDynamics.gettimes(prob)
@test times[end] ≈ tf
@test times[1] ≈ 0
@test times[2] ≈ dts[1]
@test diff(times) ≈ dts

# Test initial and final conditions
prob = Problem(model, obj, Vector(x0), tf, xf=Vector(xf))
@test prob.x0 ≈ x0
@test prob.xf ≈ xf
prob = Problem(model, obj, MVector(x0), tf, xf=MVector(xf))
@test prob.x0 ≈ x0
@test prob.xf ≈ xf
@test prob.x0 isa MVector
@test prob.xf isa MVector

x0_ = rand(n)
TO.set_initial_state!(prob, x0_)
@test prob.x0 ≈ x0_
@test_throws DimensionMismatch TO.set_initial_state!(prob, rand(2n))

## Change initial and goal states
prob = Problem(model, copy(obj), x0, tf, xf=xf, constraints=deepcopy(conSet))
x0_new = @SVector rand(n)
TO.set_initial_state!(prob, x0_new)
@test TO.get_initial_state(prob) == x0_new
TO.set_initial_state!(prob, Vector(2*x0_new))
@test TO.get_initial_state(prob) ≈ 2*x0_new

# goal state, changing both objective and terminal constraint
@test conSet[2].xf ≈ xf
xf_new = @SVector rand(n)
TO.set_goal_state!(prob, xf_new)
@test prob.xf ≈ xf_new
@test prob.obj[1].q ≈ -Q*xf_new
@test prob.constraints[2].xf ≈ xf_new

# make sure it doesn't modify the orignal since they're copied
@test obj[1].q ≈ -Q*xf
@test conSet[2].xf ≈ xf

# don't modify the terminal constraint
prob = Problem(model, copy(obj), x0, tf, xf=xf, constraints=copy(conSet))
TO.set_goal_state!(prob, xf_new, constraint=false)
@test prob.xf ≈ xf_new
@test prob.obj[1].q ≈ -Q*xf_new
@test prob.obj[end].q ≈ -Qf*xf_new
@test prob.constraints[2].xf ≈ xf

# don't modify the objective, and leave off constraints
prob = Problem(model, copy(obj), x0, tf, xf=xf)
TO.set_goal_state!(prob, xf_new, objective=false)
@test prob.xf ≈ xf_new
@test prob.obj[1].q ≈ -Q*xf
@test prob.obj[end].q ≈ -Qf*xf

# check that it modifies the orignal objective and constraint list if not copied
prob = Problem(model, obj, x0, tf, xf=xf, constraints=copy(conSet))
TO.set_goal_state!(prob, xf_new)
@test obj[1].q ≈ -Q*xf_new
@test obj[end].q ≈ -Qf*xf_new
@test conSet[2].xf ≈ xf_new

## Specify a vector of models
prob = Problem(dmodels, obj, x0, tf, xf=xf, constraints=copy(conSet))
@test prob.model === dmodels

dmodels2 = [copy(dmodel) for k = 1:N]
@test_throws AssertionError Problem(dmodels2, obj, x0, tf, xf=xf, constraints=copy(conSet))
