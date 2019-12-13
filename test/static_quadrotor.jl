prob = copy(Problems.quadrotor)
sprob = copy(Problems.quadrotor_static)
Z0 = copy(sprob.Z)
X0 = states(Z0)
U0 = controls(Z0)
prob.U ≈ U0

rollout!(prob)
rollout!(sprob)
cost(prob) ≈ cost(sprob)

# iLQR
ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

cost(prob) ≈ cost(sprob)
cost(prob)
initial_controls!(prob, U0)
solve!(prob, ilqr)
initial_controls!(silqr, U0)
solve!(silqr)

cost(prob) ≈ cost(sprob)
cost(prob)

@btime begin
    initial_controls!($prob, $U0)
    solve!($prob, $ilqr)
end

@btime begin
    initial_controls!($silqr, $U0)
    solve!($silqr)
end

# AL-iLQR
prob = copy(Problems.quadrotor)
sprob = copy(Problems.quadrotor_static)
al = AugmentedLagrangianSolver(prob)
sal = StaticALSolver(sprob)

initial_controls!(prob, U0)
solve!(prob, al)
initial_controls!(sal, U0)
solve!(sal)
max_violation(sal) ≈ max_violation(prob)
sal.solver_uncon.stats

@btime begin
    initial_controls!($prob, $U0)
    solve!($prob, $al)
end

@btime begin
    initial_controls!($sal, $U0)
    solve!($sal)
end

# Ipopt
prob = copy(Problems.quadrotor)
sprob = copy(Problems.quadrotor_static)
sprob = change_integration(sprob, HermiteSimpson)
rollout!(sprob)
ipopt = StaticDIRCOLSolver(sprob)
ipopt.opts.verbose = false
@btime solve!($ipopt)
max_violation(ipopt)
