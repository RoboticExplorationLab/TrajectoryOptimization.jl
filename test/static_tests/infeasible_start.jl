prob = copy(Problems.car_3obs_static)
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

Z = infeasible_trajectory(inf_model, Z0)
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
    allocs += @allocated jacobian(con, z)
end


con = prob.constraints[1].con
idx_con = IndexedConstraint(n,n+m,con)
@test evaluate(idx_con, z) == evaluate(con, z0)
@test jacobian(idx_con, z) == jacobian(con, z0)
@test test_con_allocs(idx_con, z) == 0

bnd = StaticBoundConstraint(n,m, u_min=[0,-2], u_max=[Inf,2])
idx_bnd = IndexedConstraint(n,n+m,bnd)

@test test_con_allocs(idx_bnd, z) == 0

evaluate(bnd, z0)
evaluate(idx_bnd, z)
@test evaluate(bnd, z0) == evaluate(idx_bnd, z)
@test [jacobian(bnd, z0) zeros(3,3)] == jacobian(idx_bnd, z)

conSet0 = copy(get_constraints(prob))
add_constraint!(conSet0, bnd, 1:N-1, 1)
conSet = change_dimension(conSet0, n, n+m)
evaluate!(conSet, Z)
jacobian!(conSet, Z)
@test size(conSet[1].∇c[1]) == (3,8)
@test size(conSet0[1].∇c[1]) == (3,5)


# Cost functions
cost0 = prob.obj[1]
ix = 1:n
iu = 1:m
idx_cost = IndexedCost(cost0,ix,iu)
@test stage_cost(cost0, x, u0) ≈ stage_cost(idx_cost, x, u)
@test stage_cost(cost0, x) ≈ stage_cost(idx_cost, x)
Qx0, Qu0 = gradient(cost0, x, u0)
Qx, Qu = gradient(idx_cost, x, u)
@test Qx0 == Qx
@test [Qu0; zeros(n)] == Qu

Qxx0, Quu0, Qux0 = hessian(cost0, x, u0)
Qxx, Quu, Qux = hessian(idx_cost, x, u)
@test Qxx0 == Qxx
@test Qux == zeros(n+m,n)
@test Quu == Diagonal([diag(Quu0); zeros(n)])

costfun = change_dimension(cost0, n, n+m)
@test stage_cost(costfun, x, u) == stage_cost(idx_cost, x, u)
@test gradient(costfun, x, u) == gradient(idx_cost, x, u)
@test hessian(costfun, x, u) == hessian(idx_cost, x, u)
@test costfun.Q isa Diagonal{Float64,<:SVector}
@test costfun.R isa Diagonal{Float64,<:SVector}
@test costfun.H isa SMatrix
@test costfun.q isa SVector
@test costfun.r isa SVector

costs = infeasible_objective(prob.obj, 2.3)
@test costs[1].Q == cost0.Q
@test costs[1].R == Diagonal([diag(cost0.R); fill(2.3,n)])
@test costs[1].q == cost0.q
@test costs[1].r == [cost0.r; zeros(n)]
@test sum(costs[1].H) == sum(cost0.H)

prob = copy(Problems.car_3obs)
sprob = copy(Problems.car_3obs_static)

al = AugmentedLagrangianSolver(prob)
sal = StaticALSolver(sprob)

solve!(prob, al)
solve!(sal)
max_violation(al)
max_violation(sal)

prob = copy(Problems.car_3obs)
altro = solve!(prob, ALTROSolverOptions{Float64}())
max_violation(prob)
max_violation(altro.solver_al)




# Step through the solve
prob = copy(Problems.car_3obs)
sprob = copy(Problems.car_3obs_static)
initial_states!(prob, X0)
initial_controls!(prob, U0)
initial_trajectory!(sprob, Z0)
states(sprob) == prob.X
controls(sprob) == prob.U

prob_inf = infeasible_problem(prob, 0.01)
sprob_inf = InfeasibleProblem(sprob, Z0, 0.01/prob.dt)
states(sprob_inf) == prob_inf.X
controls(sprob_inf) == prob_inf.U

al = AugmentedLagrangianSolver(prob_inf)
sal = StaticALSolver(sprob_inf)
Z_init = copy(sprob_inf.Z)
X_init = states(Z_init)
U_init = controls(Z_init)

solve!(prob_inf, al)
solve!(sal)
max_violation(prob_inf)
max_violation(sal)

@btime begin
    initial_controls!($prob_inf, $U_init)
    initial_states!($prob_inf, $X_init)
    solve!($prob_inf, $al)
end

@btime begin
    initial_trajectory!($sal, $Z_init)
    solve!($sal)
end
max_violation(sal)

rollout!(prob_inf)
rollout!(sprob_inf)
states(sprob_inf) == prob_inf.X

cost(prob_inf) ≈ cost(sprob_inf)
cost(prob_inf)
max_violation(prob_inf) ≈ max_violation(sprob_inf)

al = AugmentedLagrangianSolver(prob_inf)
sal = StaticALSolver(sprob_inf)
reset!(get_constraints(sal), sal.opts)

prob_al = AugmentedLagrangianProblem(prob_inf, al)
cost(sal) ≈ cost(prob_al)
silqr = sal.solver_uncon
ilqr = al.solver_uncon

# Check Dynamics Jacobians
discrete_jacobian!(silqr.∇F, silqr.model, silqr.Z)
jacobian!(prob_al, ilqr)
ilqr.∇F ≈ silqr.∇F

# Check Cost Expansion
cost_expansion(silqr.Q, silqr.obj, silqr.Z)
cost_expansion!(prob_al, ilqr)
silqr.Q.xx ≈ [Q.xx for Q in ilqr.Q]
silqr.Q.uu[1:end-1] ≈ [Q.uu for Q in ilqr.Q[1:end-1]]
silqr.Q.ux[1:end-1] ≈ [Q.ux for Q in ilqr.Q[1:end-1]]
silqr.Q.x[1:end] ≈ [Q.x for Q in ilqr.Q[1:end]]
silqr.Q.u[1:end-1] ≈ [Q.u for Q in ilqr.Q[1:end-1]]

conSet = get_constraints(sal)
prob_al.obj.constraints
conSet[3].vals ≈ [c.infeasible for c in prob_al.obj.C[1:end-1]]
conSet[3].∇c ≈ [c.infeasible[:,4:end] for c in prob_al.obj.∇C[1:end-1]]
conSet[1].vals ≈ [c.obs for c in prob_al.obj.C[2:end-1]]
conSet[1].∇c ≈ [c.obs[:,1:3] for c in prob_al.obj.∇C[2:end-1]]

Q = CostExpansion(size(prob)...)
Q.xx[1]
cost_expansion(Q,conSet[3],silqr.Z)
cost_expansion(silqr.Q, silqr.obj.obj, silqr.Z)
cost_expansion(silqr.Q, silqr.obj, silqr.Z)
cost_expansion!(prob_al, ilqr)
ilqr.Q[1].xx

silqr.Q.xx
[Q.xx for Q in ilqr.Q]
[Q.uu for Q in ilqr.Q[1:end-1]]
silqr.Q.uu[1:end-1]


solve!(prob_al, ilqr)
solve!(silqr)
cost(silqr)
cost(prob_al)


solve!(prob_inf, al)
max_violation(al)
al.stats[:iterations]

solve!(sal)
max_violation(sal)
sal.stats.iterations
al.stats.cost
