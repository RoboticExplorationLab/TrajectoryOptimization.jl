const TO = TrajectoryOptimization
using StaticArrays

include("../problems/cartpole.jl")

prob = copy(cartpole)
sprob = copy(cartpole_static)
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

@btime begin
    initial_controls!($prob, $U0)
    solve!($prob, $ilqr)
end

m = size(silqr)[2]
U0 = [SVector{m}(u) for u in U0]
@btime begin
    initial_controls!($silqr, $U0)
    solve!($silqr)
end


# Augmented Lagrangian
prob = copy(cartpole)
sprob = copy(cartpole_static)

sal = StaticALSolver(sprob)
al = AugmentedLagrangianSolver(prob)

initial_controls!(sal, U0)
solve!(sal)
max_violation(sal)

initial_controls!(prob,U0)
solve!(prob,al)
max_violation(al)
max_violation(al) ≈ max_violation(sal)

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
solve!(prob, pn)
max_violation_direct(prob)


spn = StaticPNSolver(sprob)
initial_controls!(spn, Usol)
initial_states!(spn, Xsol)
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
prob = copy(cartpole)
sprob = copy(cartpole_static)

altro = ALTROSolver(prob)
saltro = StaticALTROSolver(sprob)

opts_altro = ALTROSolverOptions(projected_newton=true)
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

X0 = [@SVector fill(NaN,n) for k = 1:N]
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
