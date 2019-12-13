using Test
prob = copy(Problems.car_3obs_static)
Z0 = copy(prob.Z)
X0 = deepcopy(states(Z0))
U0 = deepcopy(controls(Z0))

# Solve with iLQR
ilqr = StaticiLQRSolver(prob)
solve!(ilqr)
@test cost(ilqr) < 0.8182
initial_controls!(ilqr, U0)
@test (@allocated solve!(ilqr)) == 0

# Solve with AL-iLQR
al = StaticALSolver(prob)
initial_controls!(al, U0)
solve!(al)
@test max_violation(al) < al.opts.constraint_tolerance
@test cost(al) < 0.82
initial_controls!(al, U0)
@test (@allocated solve!(al)) == 0

# Solve with ALTRO
altro = StaticALTROSolver(prob)
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



# Escape problem
T = Float64
max_con_viol = 1e-3
opts_ilqr = iLQRSolverOptions()
sopts_ilqr = StaticiLQRSolverOptions()

opts_al = AugmentedLagrangianSolverOptions(
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-6,
    cost_tolerance_intermediate=1.0e-2,
    constraint_tolerance=max_con_viol,
    penalty_scaling=50.,
    penalty_initial=10.)

sopts_al = StaticALSolverOptions(
    opts_uncon=sopts_ilqr,
    cost_tolerance=1.0e-6,
    cost_tolerance_intermediate=1.0e-2,
    constraint_tolerance=max_con_viol,
    penalty_scaling=50.,
    penalty_initial=10.)

opts_altro = ALTROSolverOptions{T}(
    opts_al=opts_al,
    R_inf=1.0e-1,
    resolve_feasible_problem=false,
    opts_pn=opts_pn,
    projected_newton=false,
    projected_newton_tolerance=1.0e-3);

sopts_altro = StaticALTROSolverOptions(
    opts_al=sopts_al,
    R_inf=1.0e-1,
    resolve_feasible_problem=false,
    opts_pn=sopts_pn,
    projected_newton=false,
    projected_newton_tolerance=1.0e-3);

prob = copy(Problems.car_escape)
sprob = copy(Problems.car_escape_static)
states(sprob) == prob.X
controls(sprob) == prob.U

prob = infeasible_problem(prob, opts_altro.R_inf)
sprob = InfeasibleProblem(sprob, sprob.Z, opts_altro.R_inf/prob.dt)

al = AugmentedLagrangianSolver(prob)
sal = StaticALSolver(sprob)

solve!(prob, al)
solve!(sal)

max_violation(prob)
max_violation(sal)


# Solve with ALTRO
prob = copy(Problems.car_escape)
sprob = copy(Problems.car_escape_static)
states(sprob) == prob.X
controls(sprob) == prob.U

prob = copy(Problems.car_escape)
solve!(prob, opts_altro)
max_violation_direct(prob)

sprob = copy(Problems.car_escape_static)
altro = StaticALTROSolver(sprob, sopts_altro, infeasible=true)
Z0 = copy(get_trajectory(altro))
initial_trajectory!(altro, Z0)
altro.opts.projected_newton = false
solve!(altro)
max_violation(altro)
Z = get_trajectory(altro.solver_al)

pn = altro.solver_pn
initial_trajectory!(pn, Z)
viol0 = max_violation(pn)
reset!(pn)

solver = pn
update_constraints!(solver)
copy_constraints!(solver)
constraint_jacobian!(solver)
copy_jacobians!(solver)
cost_expansion!(solver)
update_active_set!(solver)
max_violation(pn)
norm(active_constraints(pn)[2],Inf)
max_violation(pn) == norm(active_constraints(pn)[2],Inf)
conSet = get_constraints(pn)

copyto!(solver.P, solver.Z)
copy_constraints!(solver)
copy_jacobians!(solver)
copy_active_set!(solver)

D,d = active_constraints(solver)
norm(d,Inf) == viol0
H = Diagonal(solver.H)
HinvD = H\D'
ρ = 1e-2
S = Symmetric(D*HinvD)
Sreg = cholesky(S + ρ*I)

P = solver.P
P̄ = copy(solver.P)

# line search
α = 1.0
S = (S,Sreg)
δλ = reg_solve(S[1], d, S[2], 1e-8, 25)
δZ = -HinvD*δλ
P̄.Z .= P.Z + α*δZ

copyto!(solver.Z̄, P̄)
update_constraints!(solver, solver.Z̄)
update_constraints!(solver, Z)
max_violation(solver)
findmax(map(conSet.constraints) do con
    maximum(con.c_max)
end)
findmax(maximum.(conSet[3].vals))
findmax(conSet[3].vals[71])
conSet[3].active[71][170]
conSet[3].vals[71][170]
update_active_set!(conSet, Z, Val(0.001))
pn.opts.active_set_tolerance

copy_constraints!(solver)
update_active_set!(solver)
copy_active_set!(solver)
d = active_constraints(solver)[2]
norm(d,Inf)
