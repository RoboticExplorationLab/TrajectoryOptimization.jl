import TrajectoryOptimization: InfeasibleModel, InfeasibleConstraint
const TO = TrajectoryOptimization

prob = Problems.DubinsCar(:three_obstacles)[1]
n,m,N = size(prob)
U0 = deepcopy(controls(prob))
dt = prob.Z[1].dt

# Generate an initial state trajectory
N = prob.N
x0 = prob.x0
xf = prob.xf
x_guess = SVector{N}(range(x0[1], stop=xf[1], length=N))
y_guess = SVector{N}(range(x0[2], stop=xf[2], length=N))
X0 = [SVector(x_guess[k], y_guess[k], x0[3]) for k = 1:N]
Z0 = Traj(X0,U0,dt*ones(N))

# Create infeasible model
inf_model = InfeasibleModel(prob.model)
inf_con = InfeasibleConstraint(inf_model)
Z = TO.infeasible_trajectory(inf_model, Z0)

@test states(Z) ≈ states(Z0)
rollout!(inf_model, Z, x0)
@test states(Z) ≈ states(Z0)

# Test constraints
model = prob.model
x,u0 = rand(model)
ui = @SVector rand(n)
u = [u0; ui]

z0 = KnotPoint(x,u0,dt)
z = KnotPoint(x,u,dt)

function test_con_allocs(con,z)
    allocs = @allocated evaluate(con, z)
    ∇c = TrajOptCore.gen_jacobian(con)
    allocs += @allocated jacobian!(∇c, con, z)
end


con = prob.constraints[1]
idx_con = TO.change_dimension(con, n, n+m, 1:n, 1:m)
@test evaluate(idx_con, z) == evaluate(con, z0)
∇c1 = TrajOptCore.gen_jacobian(con)
∇c2 = TrajOptCore.gen_jacobian(idx_con)
jacobian!(∇c1, con, z0)
jacobian!(∇c2, idx_con, z)
@test ∇c1 ≈ ∇c2
# @test test_con_allocs(idx_con, z) == 0

bnd = BoundConstraint(n,m, u_min=[0,-2], u_max=[Inf,2])
idx_bnd = TrajOptCore.change_dimension(bnd, n, n+m, 1:n, 1:m)

@test test_con_allocs(idx_bnd, z) == 0

@test evaluate(bnd, z0) == evaluate(idx_bnd, z)
∇c1 = TrajOptCore.gen_jacobian(bnd)
∇c2 = TrajOptCore.gen_jacobian(idx_bnd)
jacobian!(∇c1, bnd, z0)
jacobian!(∇c2, idx_bnd, z)
@test ∇c1 ≈ ∇c2[:,1:n+m]

conSet0 = copy(get_constraints(prob))
add_constraint!(conSet0, bnd, 1:N-1, 1)
conSet = TrajOptCore.change_dimension(conSet0, n, n+m, 1:n, 1:m)
@test size(conSet[1]) == (3,8)
@test size(conSet0[1]) == (3,5)


# Cost functions
cost0 = prob.obj[1]
ix = 1:n
iu = 1:m
# idx_cost = IndexedCost(cost0,ix,iu)
idx_cost = TrajOptCore.change_dimension(cost0, n, n+m, ix, iu)
@test u0 ≈ u[1:2]
@test cost0.Q ≈ idx_cost.Q
@test cost0.q ≈ idx_cost.q
@test cost0.R ≈ idx_cost.R[1:m,1:m]
@test cost0.r ≈ idx_cost.r[1:m]
@test cost0.c ≈ idx_cost.c
@test TrajOptCore.stage_cost(cost0, x, u0) ≈ TrajOptCore.stage_cost(idx_cost, x, u)
@test TrajOptCore.stage_cost(cost0, x) ≈ TrajOptCore.stage_cost(idx_cost, x)

E0 = QuadraticCost{Float64}(n,m)
E  = QuadraticCost{Float64}(n,n+m)
TrajOptCore.gradient!(E0, cost0, x, u0)
TrajOptCore.gradient!(E, idx_cost, x, u)
@test E0.q == E.q
@test [E0.r; zeros(n)] == E.r

TrajOptCore.hessian!(E0, cost0, x, u0)
TrajOptCore.hessian!(E, idx_cost, x, u)
@test E0.Q == E.Q
@test Diagonal([diag(E0.R); zeros(n)]) ≈ E.R

costs = TO.infeasible_objective(prob.obj, 2.3)
@test costs[1].Q == cost0.Q
@test costs[1].R == Diagonal([diag(cost0.R); fill(2.3,n)])
@test costs[1].q == cost0.q
@test costs[1].r == [cost0.r; zeros(n)]
# @test sum(costs[1].H) == sum(cost0.H)

# Test the solve
sprob = copy(Problems.car_3obs_static)
sal = AugmentedLagrangianSolver(sprob)
solve!(sal)
max_violation(sal)


# Test Infeasible solve
sprob = copy(Problems.car_3obs_static)
initial_trajectory!(sprob, Z0)
sprob_inf = InfeasibleProblem(sprob, Z0, 0.01/sprob.Z[1].dt)
sal = AugmentedLagrangianSolver(sprob_inf)
Z_init = copy(sprob_inf.Z)

initial_trajectory!(sal, Z_init)
solve!(sal)
max_violation(sal)

initial_trajectory!(sal, Z_init)
@test (@allocated solve!(sal)) == 0
