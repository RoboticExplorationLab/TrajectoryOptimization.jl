using Test
prob = Problems.DubinsCar(:three_obstacles)[1]
Z0 = copy(prob.Z)
X0 = deepcopy(states(Z0))
U0 = deepcopy(controls(Z0))

# Solve with iLQR
ilqr = iLQRSolver(Problems.DubinsCar(:three_obstacles)...)
Z0 = copy(ilqr.Z)
X0 = deepcopy(states(Z0))
U0 = deepcopy(controls(Z0))
solve!(ilqr)
@test cost(ilqr) < 8.47
initial_controls!(ilqr, U0)
@test (@allocated solve!(ilqr)) == 0

# Solve with AL-iLQR
al = AugmentedLagrangianSolver(Problems.DubinsCar(:three_obstacles)...)
initial_controls!(al, U0)
solve!(al)
@test max_violation(al) < al.opts.constraint_tolerance
@test cost(al) < 10
initial_controls!(al, U0)
@test (@allocated solve!(al)) == 0

# Solve with ALTRO
altro = ALTROSolver(Problems.DubinsCar(:three_obstacles)...)
altro.opts.opts_pn.verbose = false
initial_controls!(altro, U0)
solve!(altro)
Z_sol = copy(get_trajectory(altro))
X_sol = deepcopy(states(altro))
@test max_violation(altro) < 1e-8
@test cost(altro) < 10
altro.opts.projected_newton = false
initial_controls!(altro, U0)
solve!(altro)
initial_controls!(altro, U0)
@test (@allocated solve!(altro)) == 0

# Ipopt
ipopt = DIRCOLSolver(Problems.DubinsCar(:three_obstacles)..., integration=HermiteSimpson)
ipopt.opts.verbose = false
solve!(ipopt)
@test max_violation(ipopt) < 2e-8
@test cost(ipopt) < 12
