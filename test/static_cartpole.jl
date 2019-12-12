prob0 = Problems.quadrotor
sprob0 = Problems.quadrotor_static
prob0 = Problems.cartpole
sprob0 = Problems.cartpole_static
prob = copy(prob0)
sprob = copy(sprob0)
prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)
U0 = deepcopy(controls(sprob))
X0 = deepcopy(states(sprob))

ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

initial_controls!(prob, U0)
initial_states!(prob, X0)
solve!(prob, ilqr)

initial_controls!(silqr, U0)
solve!(silqr)
Xs = states(silqr)
Us = controls(silqr)
norm(Xs - prob.X)

@btime begin
    initial_controls!($prob, $U0)
    initial_states!($prob, $X0)
    solve($prob,$ilqr)
end

@btime begin
    initial_controls!($silqr, $U0)
    solve!($silqr)  # 55x speedup
end




# Now make augmented lagragian problem
prob = copy(prob0)
sprob = copy(sprob0)
U0 = deepcopy(controls(sprob))

al = AugmentedLagrangianSolver(prob)
sal = StaticALSolver(sprob)

solve!(prob, al)
max_violation(prob)
al.stats[:iterations]

solve!(sal)
max_violation(sal)
sal.stats.iterations

norm(prob.X - states(sal))
norm(prob.U - controls(sal))

@time begin
    initial_controls!(prob, U0)
    solve!(prob, al)
end
al.stats[:iterations]
max_violation(prob)

@time begin
    initial_controls!(sal, U0)
    solve!(sal)
end
sal.stats.iterations
sal.stats.cost
max_violation(sal)

@btime begin
    initial_controls!($prob, $U0)
    solve($prob,$al)
end

@btime begin
    initial_controls!($sal, $U0)
    solve!($sal)  # 65x faster!!!
end


# Projected Newton
prob = copy(prob0)
sprob = copy(sprob0)

ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

al = AugmentedLagrangianSolver(prob)
sal = StaticALSolver(sprob)

# Solve with Augmented Lagrangian
solve!(prob, al)
solve!(sal)
max_violation(al)
max_violation(sal)
states(sal) ≈ prob.X
controls(sal) ≈ prob.U

# Finish with Projected Newton
pn = ProjectedNewtonSolver(prob)
P = StaticPrimals(size(sprob)...)
spn = StaticPNSolver(sprob)

update_constraints!(prob, pn)
dynamics_constraints!(prob, pn)
update_constraints!(spn)
copy_constraints!(spn)
Vector(pn.y) .≈ spn.d

@btime begin
    update_constraints!($prob, $pn)
    dynamics_constraints!($prob, $pn)
end
@btime begin
    update_constraints!($spn)
    copy_constraints!($spn)
end  # 45x faster


dynamics_jacobian!(prob, pn)
constraint_jacobian!(prob, pn)
constraint_jacobian!(spn)
copy_jacobians!(spn)
pn.Y ≈ Matrix(spn.D)

@btime begin
    dynamics_jacobian!($prob, $pn)
    constraint_jacobian!($prob, $pn)
end
@btime begin
    constraint_jacobian!($spn)
    copy_jacobians!($spn)
end # 19x faster

active_set!(prob, pn)
update_active_set!(spn)
pn.a.duals ≈ spn.active_set
pn.a.duals ≈ spn_d.active_set

@btime update_active_set!($spn)

cost_expansion!(prob, pn)
E = CostExpansion(size(prob)...)
cost_expansion!(spn)
pn.H ≈ spn.H
pn.g ≈ spn.g

copyto!(spn.P, spn.Z)

# Test solve
ρ = 1e-2
H = Diagonal(pn.H)
Hs = Diagonal(spn.H)

Y,y = active_constraints(prob, pn)
D,d = active_constraints(spn)
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
as == a

Z = primals(pn.V)
P = spn.P
Zs = spn.Z
Z ≈ spn.P.Z

V_ = copy(pn.V)
Z_ = primals(V_)
P_ = copy(spn.P)
Zs_ = spn.Z̄

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
update_constraints!(spn, Zs_)
copy_constraints!(spn)
d = spn.d[as]
y ≈ d

norm(y,Inf)
norm(d,Inf)


# Entire PN solve
prob = copy(prob0)
sprob = copy(sprob0)

ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

al = AugmentedLagrangianSolver(prob)
prob_al = AugmentedLagrangianProblem(prob, al)

sal = StaticALSolver(sprob)

# Solve with Augmented Lagrangian
solve!(prob, al)
solve!(sal)
max_violation(al)
max_violation(sal)
states(sal) ≈ prob.X
controls(sal) ≈ prob.U

Xsol = deepcopy(prob.X)
Usol = deepcopy(prob.U)
Zsol = deepcopy(get_trajectory(sal))

# Finish with Projected Newton
pn = ProjectedNewtonSolver(prob)
spn = StaticPNSolver(sprob)

pn.opts.verbose = true
spn.opts.verbose = true

copyto!(pn.V, Xsol, Usol)
update!(prob, pn)
max_violation(pn)
solve!(prob, pn)
max_violation(pn)

initial_trajectory!(spn, Zsol)
max_violation(spn)
solve!(spn)
max_violation(spn)

pn.opts.verbose = false
spn.opts.verbose = false

@btime begin
    copyto!($pn.V, $Xsol, $Usol)
    solve!($prob, $pn)
end

@btime begin
    initial_trajectory!($spn, $Zsol)
    solve!($spn) # 8.5x faster
end


# DIRCOL
prob = copy(Problems.cartpole)
prob = update_problem(prob,model=Dynamics.cartpole)
sprob = copy(Problems.cartpole_static)
rollout!(sprob)
initial_states!(prob, state(sprob))
initial_controls!(prob, control(sprob))

# Add dynamics constraint
conSet = get_constraints(sprob)
hs_con = ConstraintVals( ExplicitDynamics{HermiteSimpson}(sprob.model, sprob.N), 1:sprob.N-1)
init_con = ConstraintVals( GoalConstraint(sprob.x0), 1:1)
add_constraint!(get_constraints(sprob), hs_con, 1)
add_constraint!(get_constraints(sprob), init_con, 1)

# Build NLP problem
bnds = remove_bounds!(prob)
dircol = DIRCOLSolver(prob)
z_U, z_L, g_U, g_L = get_bounds(prob, bnds)
d = DIRCOLProblem(prob, dircol, z_L, z_U, g_L, g_U)
ds = StaticDIRCOLProblem(sprob)

model = build_moi_problem(d)
MOI.optimize!(model)

model2 = build_moi_problem(ds)
MOI.optimize!(model2)
@btime MOI.optimize!($model)
b = @benchmark MOI.optimize!($model2) # 25x faster, 1000x less memory
