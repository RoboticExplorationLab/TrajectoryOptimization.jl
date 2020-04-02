using RobotDynamics
using TrajOptCore
using BenchmarkTools
using TrajOptCore
const TO = TrajectoryOptimization

solver = ALTROSolver(Problems.Quadrotor(:zigzag)...)
al = solver.solver_al
ilqr = al.solver_uncon

# iLQR Solve
initialize!(ilqr)
cost(ilqr.obj.obj, ilqr.Z)
cost(ilqr)
get_constraints(al)
TO.get_J(ilqr.obj)
RobotDynamics.state_diff_jacobian!(ilqr.G, ilqr.model, ilqr.Z)
RobotDynamics.dynamics_expansion!(ilqr.D, ilqr.model, ilqr.Z)
error_expansion!(ilqr.D, ilqr.model, ilqr.G)
cost_expansion!(ilqr.quad_obj, ilqr.obj, ilqr.Z)
error_expansion!(ilqr.Q, ilqr.quad_obj, ilqr.model, ilqr.Z, ilqr.G)

J_prev = cost(ilqr)
ilqr.Q[end].Q
ilqr.D[1].A
ilqr.xf
ilqr.Q[end].q
ilqr.quad_obj[end].q
ilqr.obj.obj[end].q
ΔV = TO.static_backwardpass!(ilqr)
J = TO.forwardpass!(ilqr, ΔV, J_prev)

solve!(ilqr)
cost(ilqr)

# AL-iLQR solve
TO.dual_update!(al)
TO.penalty_update!(al)
cost(al)

solve!(al)
cost(al)
max_violation(al)

# PN solve
pn = solver.solver_pn
TO.update_constraints!(pn)
TO.copy_constraints!(pn)
TO.constraint_jacobian!(pn)
TO.copy_jacobians!(pn)
TO.cost_expansion!(pn)
TO.update_active_set!(pn)
TO.copy_active_set!(pn)
count(pn.active_set)

# Assume constant, diagonal cost Hessian (for now)
H = Diagonal(pn.H)

# Update everything
TO.update_constraints!(pn)
TO.constraint_jacobian!(pn)
TO.update_active_set!(pn)
TO.cost_expansion!(pn)

# Copy results from constraint sets to sparse arrays
copyto!(pn.P, pn.Z)
TO.copy_constraints!(pn)
TO.copy_jacobians!(pn)
TO.copy_active_set!(pn)

# Get active constraints
D,d = TO.active_constraints(pn)

ρ = pn.opts.ρ
viol0 = norm(d,Inf)
HinvD = H\D'
S = Symmetric(D*HinvD)
Sreg = cholesky(S + ρ*I)

# TO._projection_linesearch!(pn, (S,Sreg), HinvD)
α = 1.0
TO.reg_solve(S, d, Sreg)
δλ = TO.reg_solve(S, d, Sreg, 1e-8, 25)
δZ = -HinvD*δλ
pn.P̄.Z .= pn.P.Z + α*δZ
copyto!(pn.Z̄, pn.P̄)
TO.update_constraints!(pn, pn.Z̄)
max_violation(pn)
TO.copy_constraints!(pn)
d = pn.d[pn.active_set]
viol = norm(d,Inf)

# Test piece-wise solve
solver = ALTROSolver(Problems.DubinsCar(:escape)..., infeasible=true, R_inf=0.1)
al = solver.solver_al
solve!(al)
iterations(al)
cost(al)

pn = solver.solver_pn
pn.opts.verbose = true
solve!(pn)
iterations(pn)
iterations(solver)

solver = ALTROSolver(Problems.DubinsCar(:escape)..., infeasible=true, R_inf=0.1)
solve!(solver)
iterations(solver)
max_violation(solver)
