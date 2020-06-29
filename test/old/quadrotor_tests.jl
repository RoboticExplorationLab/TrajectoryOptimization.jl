using Random
using StaticArrays
using LinearAlgebra
using MeshCat
using TrajOptPlots

vis = Visualizer(); open(vis);

max_con_viol = 1.0e-8
T = Float64
verbose = true

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    cost_tolerance=1e-4,
    iterations=300)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=40,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-4,
    constraint_tolerance=max_con_viol,
    penalty_scaling=10.,
    penalty_initial=1.)



# Step through the solve
model = Dynamics.Quadrotor()
n,m = 13,4

# discretization
N = 101 # number of knot points
tf = 5.0
dt = tf/(N-1) # total time

q0 = @SVector [1,0,0,0]

x0_pos = @SVector [0., -10., 1.]
x0 = [x0_pos; q0; @SVector zeros(6)]

xf = zero(x0)
xf_pos = @SVector [0., 50., 1.]
xf = [xf_pos; q0; @SVector zeros(6)]

u0 = @SVector fill(0.5*9.81/4, m)
U_hover = [copy(u0) for k = 1:N-1] # initial hovering control trajectory

# cost
Qdiag = fill(1e-3,n)
Qdiag[4:7] .= 1e-2
Q = Diagonal(SVector{13}(Qdiag))
R = (1.0e-4)*Diagonal(@SVector ones(m))
Qf_diag = 1000*Qdiag;
Qf = Diagonal(SVector{13}(Qf_diag))
obj = LQRObjective(Q,R,Qf,xf,N, uf=u0)

conSet = ConstraintSet(n,m,N)
bnd = BoundConstraint(n,m, u_min=0)
add_constraint!(conSet, bnd, 1:N-1)

# Test entire solves
model = Dynamics.Quadrotor(use_quat=false)
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
# prob = copy(Problems.quadrotor_static)
initial_controls!(prob, U_hover)
solver = iLQRSolver(prob, opts_ilqr)
solver = AugmentedLagrangianSolver(prob, opts_al)
solve!(solver)
solver.stats.iterations_total
solver.stats.iterations
cost(solver)
visualize!(vis, model, get_trajectory(solver))
plot(controls(solver))

model = Dynamics.Quadrotor(use_quat=true)
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
# prob = Problem(copy(Problems.quadrotor_static), model=model)
initial_controls!(prob, U_hover)
solver = iLQRSolver(prob, opts_ilqr)
solver = AugmentedLagrangianSolver(prob, opts_al)
solve!(solver)
solver.stats.iterations_total
solver.stats.iterations
cost(solver)
visualize!(vis, model, get_trajectory(solver))

model = Dynamics.Quadrotor(use_quat=false)
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
solver = iLQRSolver(prob, opts_ilqr)
solver = AugmentedLagrangianSolver(prob, opts_al)
solver.opts.verbose = false
@btime begin
    initial_controls!($solver, $U_hover)
    solve!($solver)
end

model = Dynamics.Quadrotor(use_quat=true)
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
solver = iLQRSolver(prob, opts_ilqr)
solver = AugmentedLagrangianSolver(prob, opts_al)
solver.opts.verbose = false
@btime begin
    initial_controls!($solver, $U_hover)
    solve!($solver)
end

plot(controls(solver))

# Solver
solver = iLQRSolver(prob, opts_ilqr)
size(solver.K[1]) == (4,13)
solver.G[1] isa LinearAlgebra.UniformScaling

J = cost(solver)
Z = solver.Z
state_diff_jacobian!(solver.G, solver.model, Z)
discrete_jacobian!(solver.∇F, solver.model, Z)
cost_expansion!(solver.Q, solver.obj, solver.Z)
solver.∇F[50]
solver.Q.x[60]
ΔV = backwardpass!(solver)
forwardpass!(solver, ΔV, J)
for k = 1:N
    solver.Z[k].z = solver.Z̄[k].z
end


# With Quaternion
model = Dynamics.Quadrotor(use_quat=true)
prob = Problem(model, obj, xf, tf, x0=x0)
initial_controls!(prob, U_hover)
rollout!(prob)
cost(prob)

solver = iLQRSolver(prob, opts_ilqr)
size(solver.K[1]) == (4,12)
solver.G[1] isa SMatrix

J = cost(solver)
Z = solver.Z
state_diff_jacobian!(solver.G, solver.model, Z)
discrete_jacobian!(solver.∇F, solver.model, Z)
cost_expansion!(solver.Q, solver.obj, solver.Z)
solver.∇F[50]
solver.Q.x[60]
ΔV = backwardpass!(solver)
forwardpass!(solver, ΔV, J)
for k = 1:N
    solver.Z[k].z = solver.Z̄[k].z
end
