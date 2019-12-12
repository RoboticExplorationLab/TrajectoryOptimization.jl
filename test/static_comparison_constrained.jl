using LinearAlgebra, StaticArrays

prob = copy(Problems.quadrotor)
sprob = copy(Problems.quadrotor_static)
U0 = deepcopy(controls(sprob))
X0 = deepcopy(states(sprob))

############################################################################################
#                               PROBLEM TESTS                                              #
############################################################################################

# Init solvers
silqr = StaticiLQRSolver(sprob)
ilqr = iLQRSolver(prob)

initial_controls!(prob, U0)
initial_states!(prob, X0)


# Test values
n,m,N = size(sprob)
xs,us = rand(sprob.model)
x,u = Vector(xs), Vector(us)
dt = prob.dt
Z = sprob.Z


# Test dynamics
@btime evaluate!($x, $(prob.model), $x, $u, $dt)
@btime discrete_dynamics($(integration(sprob)),$(sprob.model),$xs,$us,$dt) # 11x faster

# Dynamics jacobians
∇f = silqr.∇F
jacobian!(prob, ilqr)
discrete_jacobian!(∇f, sprob.model, Z)
silqr.∇F ≈ ilqr.∇F
@btime jacobian!($prob, $ilqr)
@btime discrete_jacobian!($∇f, $(sprob.model), $Z) # ≈3x faster

# Objective
cost(prob) ≈ cost(sprob)
@btime cost($prob)
@btime cost($(sprob.obj), $Z) # 7.5x faster


# Cost expansion
E = silqr.Q
cost_expansion!(prob, ilqr)
cost_expansion(E, sprob.obj, Z)
all([E.xx[k] ≈ ilqr.Q[k].xx for k in eachindex(E.xx)])
all([E.uu[k] ≈ ilqr.Q[k].uu for k = 1:N-1])
all([E.x[k] ≈ ilqr.Q[k].x for k in eachindex(E.x)])
all([E.u[k] ≈ ilqr.Q[k].u for k = 1:N-1])

@btime cost_expansion!($prob, $ilqr)
@btime cost_expansion($E, $obj, $Z)  # 7x faster


# Constraints
c_cyl = zeros(length(cyl_con))
c_sph = zeros(length(sph_con))
evaluate!(c_cyl, cyl_con, x, u)
evaluate!(c_sph, sph_con, x, u)
c_cyl ≈ evaluate(scyl_con, x, u)
c_sph ≈ evaluate(ssph_con, x, u)

@btime evaluate!($c_cyl, $cyl_con, $x, $u)
@btime evaluate($scyl_con, $xs, $us) # way faster (1000x)
@btime evaluate!($c_sph, $sph_con, $x, $u)
@btime evaluate($ssph_con, $xs, $us) # way faster (1000x)

∇c_cyl = zeros(length(cyl_con),n+m)
∇c_sph = zeros(length(sph_con),n+m)
jacobian!(∇c_cyl, cyl_con, x, u)
jacobian!(∇c_sph, sph_con, x, u)
jacobian(scyl_con, xs, us) ≈ ∇c_cyl
jacobian(ssph_con, xs, us) ≈ ∇c_sph

@btime jacobian!($∇c_cyl, $cyl_con, $x, $u)
@btime jacobian($scyl_con, $z)  # 10x faster
@btime jacobian!($∇c_sph, $sph_con, $x, $u)
@btime jacobian($ssph_con, $z) # 10x faster

b = zeros(length(bnd))
evaluate!(b, bnd, x, u)
evaluate(sbnd, xs, us) ≈ b
@btime evaluate!($b, $bnd, $x, $u)
@btime evaluate($sbnd, $xs, $us) # way faster

@btime evaluate($sgoal, $xs, $us)


# Augmented Lagrangian Solver
sopts = AugmentedLagrangianSolverOptions{Float64}()
sopts.opts_uncon = StaticiLQRSolverOptions{Float64}()
alsolver = AugmentedLagrangianSolver(prob)
salsolver = StaticALSolver(sprob, sopts)

# Augmented Lagrangian Objective
num_constraints(prob) ==  num_constraints(sprob)

alobj = AugmentedLagrangianObjective(prob, alsolver)
salobj = StaticALObjective(sprob.obj, sprob.constraints)

update_constraints!(alobj.C, alobj.constraints, X, U)
evaluate(salobj.constraints, Z)

cost!(salobj, Z)
sum(salobj.obj.J)
sum(salobj.obj.J) ≈ cost(alobj,X,U,dt_traj)

@btime cost($alobj, $X, $U, $dt_traj)
@btime cost!($salobj, $Z) # 24x faster

Q = ilqr.Q
E = silqr.Q
cost_expansion!(Q, alobj, X, U, dt_traj)
cost_expansion(E, salobj, Z)
all([E.xx[k] ≈ ilqr.Q[k].xx for k in eachindex(E.xx)])
all([E.uu[k] ≈ ilqr.Q[k].uu for k = 1:N-1])
all([E.x[k] ≈ ilqr.Q[k].x for k in eachindex(E.x)])
all([E.u[k] ≈ ilqr.Q[k].u for k = 1:N-1])
@btime cost_expansion!($Q, $alobj, $X, $U, $dt_traj)
@btime cost_expansion($E, $salobj, $Z) # 11x faster

max_violation(alsolver)
max_violation!(conSet)
maximum(conSet.c_max)
maximum(conSet.c_max) == max_violation(alsolver)
@btime max_violation($alsolver)
@btime max_violation!($conSet)  # 6.5x faster


############################################################################################
#                                TEST ENTIRE SOLVE                                         #
############################################################################################
reset!(conSet, sopts)

xs = SVector{n}(x0)
Z = [ KnotPoint(xs,us,dt) for k = 1:N]
Z[end] = KnotPoint(xs,m)
X = state.(Z)
U = control.(Z)

sprob = StaticProblem(quad_, obj, conSet, x0, xf, deepcopy(Z), deepcopy(Z), N, dt, prob.tf)

prob = copy(quad_obs)
initial_states!(prob, X)

# Augmented Lagrangian Solver
sopts = AugmentedLagrangianSolverOptions{Float64}()
sopts.opts_uncon = StaticiLQRSolverOptions{Float64}()
alsolver = AugmentedLagrangianSolver(prob)
salsolver = StaticALSolver(sprob, sopts)

prob_al = AugmentedLagrangianProblem(prob, alsolver)
sprob_al = convertProblem(sprob, salsolver)
cost(sprob_al) ≈ cost(prob_al)

ilqr = alsolver.solver_uncon
silqr = salsolver.solver_uncon

cost_expansion!(prob_al, ilqr)
cost_expansion(silqr.Q, sprob_al.obj, sprob.Z)
all([silqr.Q.xx[k] ≈ ilqr.Q[k].xx for k = 1:N])
all([silqr.Q.uu[k] ≈ ilqr.Q[k].uu for k = 1:N-1])
all([silqr.Q.x[k] ≈ ilqr.Q[k].x for k = 1:N])
all([silqr.Q.u[k] ≈ ilqr.Q[k].u for k = 1:N-1])

jacobian!(prob_al, ilqr)
discrete_jacobian!(silqr.∇F, sprob_al.model, sprob_al.Z)
ilqr.∇F ≈ silqr.∇F

ΔV  = backwardpass!(prob_al, ilqr)
ΔVs = backwardpass!(sprob_al, silqr)
ΔV ≈ ΔVs
ilqr.K ≈ silqr.K
all([state(sprob_al.Z[k]) ≈ prob_al.X[k] for k = 1:N])
all([control(sprob_al.Z[k]) ≈ prob_al.U[k] for k in 1:N-1])

rollout!(prob_al, ilqr, 1.0)
rollout!(sprob_al, silqr, 1.0)
all([state(sprob_al.Z̄[k]) ≈ ilqr.X̄[k] for k = 1:N])


reset!(silqr)
sprob = StaticProblem(quad_, obj, conSet, x0, xf, deepcopy(Z), deepcopy(Z), N, dt, prob.tf)

prob = copy(quad_obs)
initial_states!(prob, X)
initial_controls!(prob, U)

alsolver.opts.opts_uncon.iterations = 50
salsolver.opts.opts_uncon.iterations = 50

prob_al = AugmentedLagrangianProblem(prob, alsolver)
sprob_al = convertProblem(sprob, salsolver)
silqr.opts.iterations

@btime begin
    initial_controls!($prob_al, $U)
    solve!($prob_al, $ilqr)
end
ilqr.stats[:iterations]

@btime begin
    for k = 1:$N
        $sprob_al.Z[k].z = $Z[k].z
    end
    solve!($sprob_al, $silqr)  # 21x faster!!! (0 allocs)
end
silqr.stats.iterations
initial_controls!(prob_al, U)
solve!(prob_al, ilqr)
for k = 1:N
    sprob_al.Z[k].z = Z[k].z
end
solve!(sprob_al, silqr)

cost(prob_al)
cost(sprob_al)

@btime dual_update!($sprob_al, $salsolver)
@btime penalty_update!($sprob_al, $salsolver)

@btime reset!($con_bnd, $salsolver.opts)
@btime reset!($conSet, $salsolver.opts)


reset!(silqr)
sprob = StaticProblem(quad_, obj, conSet, x0, xf, deepcopy(Z), deepcopy(Z), N, dt, prob.tf)

prob = copy(quad_obs)
initial_states!(prob, X)
initial_controls!(prob, U)

alsolver.opts.opts_uncon.iterations = 50
salsolver.opts.opts_uncon.iterations = 50

prob_al = AugmentedLagrangianProblem(prob, alsolver)
sprob_al = convertProblem(sprob, salsolver)
silqr.opts.iterations
reset!(conSet, salsolver.opts)

reset!(salsolver)
reset!(salsolver.stats, 30)
salsolver.stats.c_max
salsolver.opts.iterations
@btime begin
    for k = 1:$N
        $sprob_al.Z[k].z = $Z[k].z
    end
    reset!($conSet, $salsolver.opts)
    solve!($sprob_al, $salsolver)  # 21x faster!!! (0 allocs)
end


# Test entire constrained solve
xs = SVector{n}(x0)
Z = [ KnotPoint(xs,us,dt) for k = 1:N]
Z[end] = KnotPoint(xs,m)
X = state.(Z)
U = control.(Z)

sprob = StaticProblem(quad_, obj, conSet, x0, xf, deepcopy(Z), deepcopy(Z),
    N, dt, prob.tf)

prob = copy(quad_obs)
initial_states!(prob, X)

# Augmented Lagrangian Solver
opts = AugmentedLagrangianSolverOptions{Float64}(verbose=true,
    iterations=25,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-3,
    constraint_tolerance=1e-4,
    penalty_scaling=10.,
    penalty_initial=0.1)
sopts = AugmentedLagrangianSolverOptions{Float64}(verbose=true,
    iterations=25,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-3,
    constraint_tolerance=1e-4,
    penalty_scaling=10.,
    penalty_initial=0.1)
sopts.opts_uncon = StaticiLQRSolverOptions{Float64}()
alsolver = AugmentedLagrangianSolver(prob, opts)
salsolver = StaticALSolver(sprob, sopts)


prob = copy(quad_obs)
solve!(prob, alsolver)
max_violation(prob)
maximum.(alsolver.C)
alsolver.C[end]
alsolver.stats

sprob = StaticProblem(quad_, obj, conSet, x0, xf, deepcopy(Z), deepcopy(Z),
    N, dt, prob.tf)
salsolver = StaticALSolver(sprob, sopts)
sprob_al = convertProblem(sprob, salsolver)
reset!(conSet, sopts)
solve!(sprob_al, salsolver)




function reset_problem(opts)
    xs = SVector{n}(x0)
    Z = [ KnotPoint(xs,us,dt) for k = 1:N]
    Z[end] = KnotPoint(xs,m)
    X = state.(Z)
    U = control.(Z)

    prob = copy(quad_obs)
    initial_states!(prob, X)

    # Augmented Lagrangian Solver
    opts.opts_uncon = iLQRSolverOptions{Float64}()
    alsolver = AugmentedLagrangianSolver(prob, opts)

    prob_al = AugmentedLagrangianProblem(prob, alsolver)
    return prob_al, alsolver
end

function reset_sproblem(opts)
    sopts = AugmentedLagrangianSolverOptions()
    reset!(conSet, sopts)

    xs = SVector{n}(x0)
    Z = [ KnotPoint(xs,us,dt) for k = 1:N]
    Z[end] = KnotPoint(xs,m)
    X = state.(Z)
    U = control.(Z)

    sprob = StaticProblem(quad_, obj, conSet, x0, xf, deepcopy(Z), deepcopy(Z),
        N, dt, prob.tf)

    # Augmented Lagrangian Solver
    opts.opts_uncon = StaticiLQRSolverOptions{Float64}()
    salsolver = StaticALSolver(sprob, opts)

    sprob_al = convertProblem(sprob, salsolver)
    return sprob_al, salsolver
end

function copy_traj(prob::StaticProblem, solver)
    for k = 1:N
        sprob_al.Z[k].z = sprob_al.Z̄[k].z
    end
end

function copy_traj(prob::Problem, solver)
    copyto!(prob_al.X, ilqr.X̄)
    copyto!(prob_al.U, ilqr.Ū)
end


function take_steps(prob, solver, steps=1)
    J_prev = cost(prob)
    for i = 1:steps
        J = step!(prob, solver, J_prev)
        copy_traj(prob, solver)
        dJ = J_prev - J
        # println("   iter $i: J = $J, dJ = $dJ")
        J_prev = J
    end
    return J_prev
end

opts = AugmentedLagrangianSolverOptions{Float64}(verbose=true,
    iterations=25,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-3,
    constraint_tolerance=1e-4,
    penalty_scaling=10.,
    penalty_initial=0.1)
prob_al, alsolver = reset_problems(opts)
sprob_al, salsolver = reset_sproblem(opts)

ilqr = alsolver.solver_uncon
silqr = salsolver.solver_uncon

take_steps(prob_al, ilqr, 50)
take_steps(sprob_al, silqr, 50)

dual_update!(prob_al, alsolver)
penalty_update!(prob_al, alsolver)

dual_update!(sprob_al, salsolver)
penalty_update!(sprob_al, salsolver)

max_violation(sprob_al)
max_violation(alsolver)

# Normal Step
jacobian!(prob_al,ilqr)
cost_expansion!(prob_al,ilqr)
δv = backwardpass!(prob_al,ilqr)
J = forwardpass!(prob_al,ilqr,δv,J_prev)

# Static step
discrete_jacobian!(silqr.∇F, sprob_al.model, sprob_al.Z)
cost_expansion(silqr.Q, sprob_al.obj, sprob_al.Z)
ΔV = backwardpass!(sprob_al, silqr)
Js = forwardpass!(sprob_al, silqr, ΔV, J_prev)

α = 1.0
rollout!(prob_al, ilqr, α)
rollout!(sprob_al, silqr, α)
@btime rollout!($sprob_al, $silqr, $α)
cost!(sprob_al.obj, sprob_al.Z̄)
_Js = sum( get_J(sprob_al.obj) )
_J = cost(prob_al.obj, ilqr.X̄, ilqr.Ū, get_dt_traj(prob_al))
_Js - _J
α /= 2

all([state(sprob_al.Z̄[k]) ≈ ilqr.X̄[k] for k = 1:N])
all([control(sprob_al.Z̄[k]) ≈ ilqr.Ū[k] for k in 1:N-1])


J = forwardpass!(prob_al,ilqr,δv,J_prev)
Js = forwardpass!(sprob_al, silqr, ΔV, J_prev)


all([silqr.Q.xx[k] ≈ ilqr.Q[k].xx for k = 1:N])
all([silqr.Q.uu[k] ≈ ilqr.Q[k].uu for k = 1:N-1])
all([silqr.Q.x[k] ≈ ilqr.Q[k].x for k = 1:N])
all([silqr.Q.u[k] ≈ ilqr.Q[k].u for k = 1:N-1])
ilqr.∇F ≈ silqr.∇F

δv ≈ ΔV
ilqr.K ≈ silqr.K
ilqr.d ≈ silqr.d
all([state(sprob_al.Z[k]) ≈ prob_al.X[k] for k = 1:N])
all([control(sprob_al.Z[k]) ≈ prob_al.U[k] for k in 1:N-1])


J_prev = forwardpass!(sprob_al, silqr, ΔV, J_prev)
forwardpass!(prob_al, ilqr, ΔV, J_prev) ≈ J_prev
