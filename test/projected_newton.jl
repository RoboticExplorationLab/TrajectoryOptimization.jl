using TrajectoryOptimization
using LinearAlgebra
using TrajOptCore
using RobotDynamics
using BenchmarkTools
using StaticArrays
using Test
const TO = TrajectoryOptimization

# Solve AL-iLQR problem
prob = Problems.Cartpole()[1]
prob.constraints.p
solver = ALTROSolver(Problems.Cartpole()...)
al = solver.solver_al
solve!(al)
@test iterations(al) == 39
max_violation(al)

# pn = solver.solver_pn
# pn.opts.verbose = true
# solve!(pn)

# Test that the constraint sets are "linked"
pn = solver.solver_pn
conSet0 = get_constraints(al)
conSet = get_constraints(pn)
@test conSet.λ[3] === conSet0.λ[1]
@test conSet.μ[3] === conSet0.μ[1]
@test conSet.convals[4] === conSet0.convals[2]

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
@test max_violation(pn) < viol0
TO.copy_constraints!(pn)
d = pn.d[pn.active_set]
viol = norm(d,Inf)
@test viol ≈ max_violation(pn)
