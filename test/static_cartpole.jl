prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)
u0 = control(sprob)[1]

ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

solve!(prob, ilqr)

solve!(sprob, silqr)
Xs = state(sprob)
Us = control(sprob)
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

b = @benchmark begin
    for k = 1:$sprob.N
        $sprob_al.Z[k].z = [$x0*NaN; $u0]
    end
    reset!($sprob_al.obj.constraints, $sal.opts)
    solve!($sprob_al, $sal)  # 65x faster!!!
end
@test allocs(maximum(b)) == 0


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

# Update problem with dynamics constraint
sprob_d = copy(sprob_al)
add_dynamics_constraints!(sprob_d)
@test num_constraints(sprob_d)[1] == 10
@test num_constraints(sprob_d)[2] == 6
@test num_constraints(sprob_al)[1] == 2
@test num_constraints(sprob)[1] == 2

# Finish with Projected Newton
pn = ProjectedNewtonSolver(prob)
P = StaticPrimals(size(sprob)...)
spn = StaticPNSolver(sprob_al)
spn_d = StaticPNSolver(sprob_d)

update_constraints!(prob, pn)
dynamics_constraints!(prob, pn)
update_constraints!(sprob_al, spn)
copy_constraints!(sprob_al, spn)
Vector(pn.y) ≈ spn.d

update_constraints!(sprob_d, spn_d)
copy_constraints!(sprob_d, spn_d)
spn.d ≈ spn_d.d

@btime begin
    update_constraints!($prob, $pn)
    dynamics_constraints!($prob, $pn)
end
@btime begin
    update_constraints!($sprob_al, $spn)
    copy_constraints!($sprob_al, $spn)
end  # 45x faster
@btime begin
    update_constraints!($sprob_d, $spn_d)
    copy_constraints!($sprob_d, $spn_d)
end  # 45x faster


dynamics_jacobian!(prob, pn)
constraint_jacobian!(prob, pn)
constraint_jacobian!(sprob_al, spn)
copy_jacobians!(sprob_al, spn)
pn.Y ≈ Matrix(spn.D)

constraint_jacobian!(sprob_d, spn_d)
copy_jacobians!(sprob_d, spn_d)
spn.D ≈ spn_d.D


@btime begin
    dynamics_jacobian!($prob, $pn)
    constraint_jacobian!($prob, $pn)
end
@btime begin
    constraint_jacobian!($sprob_al, $spn)
    copy_jacobians!($sprob_al, $spn)
end # 19x faster
@btime begin
    constraint_jacobian!($sprob_d, $spn_d)
    copy_jacobians!($sprob_d, $spn_d)
end # 19x faster

active_set!(prob, pn)
update_active_set!(sprob_al, spn)
update_active_set!(sprob_d, spn_d)
pn.a.duals ≈ spn.active_set
pn.a.duals ≈ spn_d.active_set

@btime update_active_set!($sprob_al, $spn)

cost_expansion!(prob, pn)
E = CostExpansion(size(prob)...)
cost_expansion!(sprob_al, spn)
cost_expansion!(sprob_d, spn_d)
pn.H ≈ spn.H
pn.g ≈ spn.g
pn.H ≈ spn_d.H
pn.g ≈ spn_d.g

copyto!(spn.P, sprob_al.Z)
copyto!(spn_d.P, sprob_d.Z)

# Test solve
ρ = 1e-2
H = Diagonal(pn.H)
Hs = Diagonal(spn.H)
Hs_ = Diagonal(spn_d.H)

Y,y = active_constraints(prob, pn)
D,d = active_constraints(sprob_d, spn_d)
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
as = spn_d.active_set
as == a

Z = primals(pn.V)
P = spn.P
Zs = sprob_d.Z
Z ≈ spn_d.P.Z

V_ = copy(pn.V)
Z_ = primals(V_)
P_ = copy(spn_d.P)
Zs_ = sprob_d.Z̄

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
update_constraints!(sprob_d, spn_d, Zs_)
copy_constraints!(sprob_d, spn_d)
d = spn_d.d[as]
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

# Add in dynamics constraints
sprob_d = copy(sprob_al)
add_dynamics_constraints!(sprob_d)

# Finish with Projected Newton
pn = ProjectedNewtonSolver(prob)
P = StaticPrimals(size(sprob)...)
spn = StaticPNSolver(sprob_d)

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
