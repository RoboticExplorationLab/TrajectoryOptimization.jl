# Escape problem
T = Float64
max_con_viol = 1e-4
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

prob = infeasible_problem(prob, opts_altro.R_inf)
sprob = InfeasibleProblem(sprob, sprob.Z, opts_altro.R_inf/prob.dt)

# Solve with augmented lagrangian
al = AugmentedLagrangianSolver(prob, opts_al)
sal = StaticALSolver(sprob, sopts_al)

solve!(prob, al)
solve!(sal)

max_violation(prob)
max_violation(sal)

# Solve with ALTRO
prob = copy(Problems.car_escape)
sprob = copy(Problems.car_escape_static)
Z0 = copy(sprob.Z)
X0 = states(Z0)
U0 = controls(Z0)

opts_altro.projected_newton = true
initial_controls!(prob, U0)
initial_states!(prob, X0)
solve!(prob, opts_altro)
max_violation(prob)

saltro = StaticALTROSolver(sprob, sopts_altro, infeasible=true)
Z_init = copy(get_trajectory(saltro))
saltro.opts.projected_newton = true
initial_trajectory!(saltro, Z_init)
solve!(saltro)
max_violation(saltro)

@btime begin
    initial_controls!($prob, $U0)
    initial_states!($prob, $X0)
    solve!($prob, $opts_altro)
end

@btime begin
    initial_trajectory!($saltro, $Z_init)
    solve!($saltro)  # 87x faster!
end



# Create Solvers
# pn = ProjectedNewtonSolver(prob)
# spn = StaticPNSolver(sprob)
pn = altro.solver_pn
spn = saltro.solver_pn

pn.V = PrimalDual(prob)
copyto!(spn.P, spn.Z)
pn.V.Z ≈ spn.P.Z

# Update everything
V = pn.V
dynamics_constraints!(prob, pn, V)
update_constraints!(prob, pn, V)
update_constraints!(spn)
copy_constraints!(spn)
spn.d ≈ pn.y

dynamics_jacobian!(prob, pn, V)
constraint_jacobian!(prob, pn, V)
constraint_jacobian!(spn)
copy_jacobians!(spn)
spn.D ≈ pn.Y

active_set!(prob, pn)
update_active_set!(spn)
copy_active_set!(spn)
spn.active_set == pn.a.duals

Y,y = active_constraints(prob, pn)
D,d = active_constraints(spn)
Y ≈ D
y ≈ d

cost_expansion!(prob, pn, V)
cost_expansion!(spn)
spn.H == pn.H
H = Diagonal(spn.H)

# Compute Search Direction
ρ = 1e-2

HinvD = H\D'
HinvY = H\Y'

Sd = Symmetric(D*HinvD)
Sy = Symmetric(Y*HinvY)
Sd ≈ Sy

Sd_reg = cholesky(Sd + ρ*I)
Sy_reg = cholesky(Sy + ρ*I)

norm(spn.d[spn.active_set],Inf)
_projection_linesearch!(prob, pn, V, (Sy, Sy_reg), HinvY)
_projection_linesearch!(spn, (Sd, Sd_reg), HinvD)
