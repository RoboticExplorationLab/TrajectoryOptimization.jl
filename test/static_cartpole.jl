prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)
u0 = control(sprob)[1]

ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

solve!(prob, ilqr)
plot(prob.X)


solve!(sprob, silqr)
Xs = state(sprob)
Us = control(sprob)
plot(Xs)
norm(Xs - prob.X)

U0 = [u0 for k = 1:prob.N-1]
@btime begin
    initial_controls!($prob, $U0)
    solve($prob,$ilqr)
end

x0 = sprob.x0
@btime begin
    for k = 1:$sprob.N
        $sprob.Z[k].z = [$x0*NaN; $u0]
    end
    solve!($sprob, $silqr)  # 55x speedup
end


# Now make augmented lagragian problem
prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)

al = AugmentedLagrangianSolver(prob)
prob_al = AugmentedLagrangianProblem(prob, al)

sopts = AugmentedLagrangianSolverOptions{Float64}()
sopts.opts_uncon = StaticiLQRSolverOptions()
reset!(sprob.constraints, sopts)
sal = StaticALSolver(sprob, sopts)
sprob_al = convertProblem(sprob, sal)

solve!(prob, al)
max_violation(prob)

solve!(sprob_al, sal)
max_violation(sprob_al)



U0 = [Vector(u0) for k = 1:prob.N-1]
@time begin
    initial_controls!(prob, U0)
    solve!(prob, al)
end
al.stats[:iterations]
max_violation(prob)

x0 = sprob.x0
u0 = control(sprob)[1]
@time begin
    for k = 1:sprob.N
        sprob_al.Z[k].z = [x0*NaN; u0]
    end
    reset!(sprob_al.obj.constraints, sopts)

    solve!(sprob_al, sal)  # 55x speedup
end
sal.stats.iterations
max_violation(sprob_al)

@btime begin
    initial_controls!($prob, $U0)
    solve($prob,$al)
end

@btime begin
    for k = 1:$sprob.N
        $sprob_al.Z[k].z = [$x0*NaN; $u0]
    end
    reset!($sprob_al.obj.constraints, $sal.opts)
    solve!($sprob_al, $sal)  # 65x faster!!!
end


ilqr = iLQRSolver(prob_al)
silqr = StaticiLQRSolver(sprob_al)

solve!(prob_al, ilqr)
solve!(sprob_al, silqr)
norm(state(sprob_al) - prob_al.X)

for i = 1:10
    dual_update!(prob_al, al)
    penalty_update!(prob_al, al)

    dual_update!(sprob_al, sal)
    penalty_update!(sprob_al, sal)

    solve!(prob_al, ilqr)
    solve!(sprob_al, silqr)
end
norm(state(sprob_al) - prob_al.X)

ilqr.ρ[1]
silqr.ρ[1]

max_violation(al)
max_violation(sprob_al)


# Projected Newton
prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)

ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

al = AugmentedLagrangianSolver(prob)
prob_al = AugmentedLagrangianProblem(prob, al)

sopts = AugmentedLagrangianSolverOptions{Float64}()
sopts.opts_uncon = StaticiLQRSolverOptions()
reset!(sprob.constraints, sopts)
sal = StaticALSolver(sprob, sopts)
sprob_al = convertProblem(sprob, sal)

# Solve with Augmented Lagrangian
solve!(prob, al)
solve!(sprob_al, sal)
max_violation(al)
max_violation(sprob_al)
state(sprob_al) ≈ prob.X
control(sprob_al) ≈ prob.U

# Finish with Projected Newton
pn = ProjectedNewtonSolver(prob)
P = StaticPrimals(size(sprob)...)
spn = StaticPNSolver(sprob_al)

update_constraints!(prob, pn)
dynamics_constraints!(prob, pn)
update_constraints!(sprob_al, spn)
copy_constraints!(sprob_al, spn)
Vector(pn.y) ≈ spn.d

@btime begin
    update_constraints!($prob, $pn)
    dynamics_constraints!($prob, $pn)
end
@btime begin
    update_constraints!($sprob_al, $spn)
    copy_constraints!($sprob_al, $spn)
end  # 45x faster


dynamics_jacobian!(prob, pn)
constraint_jacobian!(prob, pn)
constraint_jacobian!(sprob_al, spn)
copy_jacobians!(sprob_al, spn)
pn.Y ≈ Matrix(spn.D)
@btime begin
    dynamics_jacobian!($prob, $pn)
    constraint_jacobian!($prob, $pn)
end
@btime begin
    constraint_jacobian!($sprob_al, $spn)
    copy_jacobians!($sprob_al, $spn)
end # 19x faster

active_set!(prob, pn)
update_active_set!(sprob_al, spn)
pn.a.duals ≈ spn.active_set

cost_expansion!(prob, pn)
E = CostExpansion(size(prob)...)
cost_expansion(E, sprob_al, spn)
pn.H ≈ spn.H
pn.g ≈ spn.g

copyto!(spn.P, sprob_al.Z)

# Test solve
ρ = 1e-2
H = Diagonal(pn.H)
Hs = Diagonal(spn.H)

Y,y = active_constraints(prob, pn)
D,d = active_constraints(sprob_al, spn)
Y ≈ D
y ≈ d

HinvY = H\Y'
HinvD = Hs\D'

S = Symmetric(Y*HinvY)
Ss = Symmetric(D*HinvD)
S ≈ Ss

Sreg = cholesky(S + ρ*I)
Ssreg = cholesky(Ss + ρ*I)


# Linesearch
a = pn.a.duals
as = spn.active_set

Z = primals(pn.V)
P = spn.P
Zs = sprob_al.Z
Z ≈ spn.P.Z

V_ = copy(pn.V)
Z_ = primals(V_)
P_ = copy(spn.P)
Zs_ = sprob_al.Z̄

α = 1.0
ϕ = 0.5
cnt = 1

δλ = reg_solve(S, y, Sreg, 1e-8, 25)
δλs = reg_solve(Ss, d, Ssreg, 1e-8, 25)
δλ ≈ δλs

δZ = -HinvY*δλ
δZs = -HinvD*δλs
δZ ≈ δZs

Z_ .= Z + α*δZ
P_.Z .= P.Z + α*δZs
Z_ ≈ P_.Z

dynamics_constraints!(prob, pn, V_)
update_constraints!(prob, pn, V_)
y = pn.y[a]

copyto!(Zs_, P_)
update_constraints!(sprob_al, spn, Zs_)
copy_constraints!(sprob_al, spn)
d = spn.d[as]
y ≈ d

norm(y,Inf)
norm(d,Inf)


# Entire PN solve
prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)

ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

al = AugmentedLagrangianSolver(prob)
prob_al = AugmentedLagrangianProblem(prob, al)

sopts = AugmentedLagrangianSolverOptions{Float64}()
sopts.opts_uncon = StaticiLQRSolverOptions()
reset!(sprob.constraints, sopts)
sal = StaticALSolver(sprob, sopts)
sprob_al = convertProblem(sprob, sal)

# Solve with Augmented Lagrangian
solve!(prob, al)
solve!(sprob_al, sal)
max_violation(al)
max_violation(sprob_al)
state(sprob_al) ≈ prob.X
control(sprob_al) ≈ prob.U

Xsol = deepcopy(prob.X)
Usol = deepcopy(prob.U)
Zsol = deepcopy(sprob_al.Z)

# Finish with Projected Newton
pn = ProjectedNewtonSolver(prob)
P = StaticPrimals(size(sprob)...)
spn = StaticPNSolver(sprob_al)

pn.opts.verbose = true
spn.opts.verbose = true

copyto!(pn.V, Xsol, Usol)
update!(prob, pn)
max_violation(pn)
solve!(prob, pn)
max_violation(pn)

initial_trajectory!(sprob_al, Zsol)
solve!(sprob_al, spn)

pn.opts.verbose = false
spn.opts.verbose = false

@btime begin
    copyto!($pn.V, $Xsol, $Usol)
    solve!($prob, $pn)
end

@btime begin
    initial_trajectory!($sprob_al, $Zsol)
    solve!($sprob_al, $spn) # 8.5x faster
end
