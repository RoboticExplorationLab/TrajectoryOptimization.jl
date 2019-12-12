const TO = TrajectoryOptimization
using BenchmarkTools
using StaticArrays, LinearAlgebra


prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)
U0 = deepcopy(prob.U)

ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

initial_controls!(prob, U0)
solve!(prob, ilqr)
ilqr.stats[:iterations]

initial_controls!(silqr, U0)
solve!(silqr)
silqr.stats.iterations
norm(prob.X - states(silqr))

m = size(silqr)[2]
U0 = [SVector{m}(u) for u in U0]
initial_controls!(silqr, U0)
@test (@allocated solve!(silqr)) == 0

@btime begin
    initial_controls!($prob, $U0)
    solve!($prob, $ilqr)
end

@btime begin
    initial_controls!($silqr, $U0)
    solve!($silqr)
end


# Augmented Lagrangian
prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)

sal = StaticALSolver(sprob)
al = AugmentedLagrangianSolver(prob)

initial_controls!(sal, U0)
solve!(sal)
max_violation(sal)

initial_controls!(prob,U0)
solve!(prob,al)
max_violation(al)
max_violation(al) ≈ max_violation(sal)

initial_controls!(sal, U0)
@test (@allocated solve!(sal)) == 0

@btime begin
    initial_controls!($prob,$U0)
    solve!($prob,$al)
end

@btime begin
    initial_controls!($sal, $U0)
    solve!($sal)
end


# Projected Newton
Xsol = deepcopy(states(sal))
Usol = deepcopy(controls(sal))
Xsol ≈ prob.X
Usol ≈ prob.U

pn = ProjectedNewtonSolver(prob)

initial_controls!(prob, Usol)
initial_states!(prob, Xsol)
copyto!(pn.V, Xsol, Usol)
max_violation(pn)
solve!(prob, pn)
max_violation_direct(prob)


spn = StaticPNSolver(sprob)
initial_controls!(spn, Usol)
initial_states!(spn, Xsol)
max_violation(spn)
solve!(spn)
max_violation(spn)

pn.opts.verbose = false
spn.opts.verbose = false

@btime begin
    copyto!($pn.V, $Xsol, $Usol)
    solve!($prob, $pn)
end

@btime begin
    initial_controls!($spn, $Usol)
    initial_states!($spn, $Xsol)
    solve!($spn)
end


# ALTRO
prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)
U0 = deepcopy(controls(sprob))

saltro = StaticALTROSolver(sprob)

opts_altro = ALTROSolverOptions{Float64}(projected_newton=true)
initial_controls!(prob, U0)
solve!(prob, opts_altro)
max_violation_direct(prob)

saltro.opts.projected_newton = true
initial_controls!(saltro, U0)
solve!(saltro)
get_trajectory(saltro)
norm(states(saltro.solver_pn) - prob.X)
max_violation(saltro)
max_violation(sprob)
TO.max_violation_direct(prob)

X0 = [@SVector fill(NaN,size(prob)[1]) for k = 1:prob.N]
opts_altro.opts_pn.verbose = false
@btime begin
    initial_controls!($prob, $U0)
    initial_states!($prob, $X0)
    solve!($prob, $opts_altro)
end

saltro.opts.opts_pn.verbose = false
@btime begin
    initial_controls!($saltro, $U0)
    solve!($saltro)  # 70x faster!!
end

# DIRCOL
prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)
rollout!(prob)
rollout!(sprob)

U0 = deepcopy(prob.U)
sprob = change_integration(sprob, HermiteSimpson)

ds = StaticDIRCOLSolver(sprob)

# Test initial state and control setters
n,m = size(ds)
u3 = ds.optimizer.variable_info[3(n+m)]
x4 = ds.optimizer.variable_info[3(n+m)+1]
@test u3.start == prob.U[3][1]
@test x4.start == prob.X[4][1]

X1 = [rand(n) for k = 1:prob.N]
U1 = [ones(m) for k = 1:prob.N-1]
initial_controls!(ds,U1)
@test u3.start == U1[3][1]
initial_states!(ds,X1)
@test x4.start == X1[4][1]

grad_f = zeros(ds.NN)
cost(ds)
cost_gradient!(ds)
copy_gradient!(grad_f, ds)
@test (@allocated cost(ds)) == 16
@test (@allocated cost_gradient!(ds)) == 0
@test (@allocated copy_gradient!(grad_f, ds)) == 0

jac_struct = MOI.jacobian_structure(ds)
g = zeros(ds.NP)
jac = zeros(length(jac_struct))

update_constraints!(ds)
constraint_jacobian!(ds)
copy_constraints!(g,ds)
copy_jacobians!(jac,ds)
@test (@allocated update_constraints!(ds)) == 0
@test (@allocated constraint_jacobian!(ds)) == 0
@test (@allocated copy_constraints!(g, ds)) == 0
@test (@allocated copy_jacobians!(jac, ds)) == 0

@btime solve!($ds)
