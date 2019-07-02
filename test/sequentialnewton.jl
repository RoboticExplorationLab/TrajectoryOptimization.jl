import TrajectoryOptimization: primals, duals, packZ, update!

# Create solver
opts = ProjectedNewtonSolverOptions{Float64}()
solver = SequentialNewtonSolver(prob,opts)
NN = N*n + (N-1)*m

# Test update functions
TO.dynamics_constraints!(prob, solver)
TO.dynamics_jacobian!(prob, solver)
TO.update_constraints!(prob, solver)
TO.constraint_jacobian!(prob, solver)
TO.active_set!(prob, solver)
TO.cost_expansion!(prob, solver)
TO.invert_hessian!(prob, solver)

# Init solvers
solver0 = ProjectedNewtonSolver(prob,opts)
TO.update!(prob, solver0)
solver = SequentialNewtonSolver(prob, opts)
TO.update!(prob, solver)

# Projection
Hinv = inv(Diagonal(solver0.H))
Y,y0 = TO.active_constraints(prob, solver0)
δz0 = -Hinv*Y'*((Y*Hinv*Y')\y0)

@test packZ(TO._projection(solver)[1:2]...) ≈ δz0

# Multiplier projection
λ0 = duals(solver0.V)[solver0.a.duals]
res0 = solver0.g + Y'λ0

λa = TO.active_duals(solver.V, solver.active_set)
δλ = TO._mult_projection(solver, λa)
@test vcat(δλ...) ≈ -(Y*Y')\(Y*res0)

# KKT solve
S0 = Symmetric(Y*Hinv*Y')
r = solver0.g + Y'λ0
δλ0 = cholesky(S0)\(y0 - Y*Hinv*r)
δz0 = -Hinv*(r + Y'δλ0)
δV1,δλ1 = TO.solveKKT_Shur(prob,solver0,Hinv)

δx, δu, δλ = TO._solveKKT(solver, λa)
@test packZ(δx, δu) ≈ δz0
@test packZ(δx, δu) ≈ δV1[1:NN]
@test vcat(δλ...) ≈ δλ0
@test vcat(δλ...) ≈ δλ1


# Update methods
δV = zero(solver0.V.V)
δV[solver0.parts.primals] .= δz0
δV[solver0.parts.duals[solver0.a.duals]] .= δλ0
V_0 = solver0.V + δV

V_0 = solver0.V + δV
V_ = copy(solver.V)
TO._update_primals!(V_, δx, δu)
TO._update_duals!(V_, δλ, solver.active_set)
@test V_.X ≈ V_0.X
@test V_.U ≈ V_0.U
@test vcat(V_.λ...) ≈ V_0.Y

V2_ = solver.V + (δx, δu)
@test V2_.X == V_.X
@test V2_.U == V_.U
@test V2_.λ == solver.V.λ

V2_ = solver.V + (δλ, solver.active_set)
@test V2_.λ == V_.λ

V2_ = solver.V + ((δx, δu, δλ), solver.active_set)
@test V2_.X == V_.X
@test V2_.U == V_.U
@test V2_.λ == V_.λ



# Newton Step from beginning
opts.verbose = false
solver0 = ProjectedNewtonSolver(prob, opts)
update!(prob, solver0)
solver = SequentialNewtonSolver(prob, opts)
update!(prob, solver)
V = solver.V

# Projection
TO.projection!(prob, solver0)
TO.projection!(prob, solver, V)

# Multiplier projection
norm(TO.residual(prob,solver0))
r0, δλ0 = TO.multiplier_projection!(prob, solver0)
TO.res2(solver, V)
r, δλ = TO.multiplier_projection!(solver, V)
@test r ≈ r0
@test vcat(δλ...) ≈ δλ0

# KKT Solve
J0 = cost(prob, solver0.V)
viol0 = max_violation(solver0)
Y,y0 = TO.active_constraints(prob, solver0)
λ0 = duals(solver0.V)[solver0.a.duals]
δV0,δλ0 = TO.solveKKT_Shur(prob, solver0, Hinv)

J = cost(prob, V)
viol = max_violation(solver)
λa = TO.active_duals(V, solver.active_set)
δx, δu, δλ = TO.solveKKT(solver, V)

@test J0 ≈ J
@test viol0 ≈ viol
@test packZ(δx, δu) ≈ δV0[1:NN]
@test norm(vcat(δλ...) - δλ0) < 1e-10

res_init = TO.res2(solver,V)
@test norm(TO.residual(prob,solver0)) ≈ res_init

# Line Search
α = 0.9
V_0 = solver0.V + α*δV0
V_ = V + (α.*(δx,δu,δλ),solver.active_set)

@test V_0.X ≈ V_.X
@test V_0.U ≈ V_.U
@test V_0.Y ≈ vcat(V_.λ...)

TO.projection!(prob, solver0, V_0)
TO.projection!(prob, solver, V_)
res0, = TO.multiplier_projection!(prob, solver0, V_0)
res, = TO.multiplier_projection!(solver, V_)
@test res0 ≈ res
J0 = cost(prob, V_0)
J = cost(prob, V_)
@test J0 ≈ J

TO.line_search(prob, solver0, δV0)
TO.line_search(prob, solver, δx, δu, δλ)


# Test Whole Step
opts = ProjectedNewtonSolverOptions{Float64}(feasibility_tolerance=1e-10)
opts.verbose = false

solver0 = ProjectedNewtonSolver(prob,opts)
begin
    copyto!.(solver0.V.X, prob.X)
    copyto!.(solver0.V.U, prob.U)
    TO.newton_step!(prob, solver0)
end

solver = SequentialNewtonSolver(prob, opts)
begin
    copyto!(solver.V.X, prob.X)
    copyto!(solver.V.U, prob.U)
    TO.newton_step!(prob, solver)
end
