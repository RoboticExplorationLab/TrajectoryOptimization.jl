using Test
prob = copy(Problems.car_3obs_static)
Z0 = copy(prob.Z)
X0 = deepcopy(states(Z0))
U0 = deepcopy(controls(Z0))

# Solve with iLQR
ilqr = iLQRSolver(prob)
solve!(ilqr)
@test cost(ilqr) < 0.8182
initial_controls!(ilqr, U0)
@test (@allocated solve!(ilqr)) == 0

# Solve with AL-iLQR
al = AugmentedLagrangianSolver(prob)
initial_controls!(al, U0)
solve!(al)
@test max_violation(al) < al.opts.constraint_tolerance
@test cost(al) < 0.82
initial_controls!(al, U0)
@test (@allocated solve!(al)) == 0

# Solve with ALTRO
altro = ALTROSolver(prob)
altro.opts.opts_pn.verbose = false
initial_controls!(altro, U0)
solve!(altro)
Z_sol = copy(get_trajectory(altro))
X_sol = deepcopy(states(altro))
@test max_violation(altro) < 1e-8
@test cost(altro) < 0.82
altro.opts.projected_newton = false
initial_controls!(altro, U0)
solve!(altro)
initial_controls!(altro, U0)
@test (@allocated solve!(altro)) == 0

# Ipopt
prob_ipopt = copy(Problems.car_3obs_static)
prob_ipopt = change_integration(prob_ipopt, HermiteSimpson)
rollout!(prob_ipopt)
ipopt = DIRCOLSolver(prob_ipopt)
ipopt.opts.verbose = false
solve!(ipopt)
@test max_violation(ipopt) < 2e-8
@test cost(ipopt) < 1.06
