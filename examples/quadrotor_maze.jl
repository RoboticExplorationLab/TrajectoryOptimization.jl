
using TrajectoryOptimization
using Random
using StaticArrays
using LinearAlgebra
using MeshCat
using TrajOptPlots
using Rotations
using BenchmarkTools
import TrajectoryOptimization: Rotation
const TO = TrajectoryOptimization

max_con_viol = 1.0e-3
T = Float64
verbose = true

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    iterations=300)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=40,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-4,
    constraint_tolerance=max_con_viol,
    penalty_scaling=10.,
    penalty_initial=1.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol,
    solve_type=:feasible)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    R_inf=1.0e-8,
    resolve_feasible_problem=false,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-4)


model = Dynamics.Quadrotor2{UnitQuaternion{Float64,IdentityMap}}(use_rot=false)
n,m = 13,4

if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end



Rot = MRP{Float64}
prob = Problems.QuadrotorMaze(Rot,use_rot=false)
opts_ilqr.verbose = false
opts_altro.projected_newton = false
solver = ALTROSolver(prob, opts_altro, infeasible=true)
Z0 = deepcopy(get_trajectory(solver))
visualize!(vis, solver)

initial_trajectory!(solver, Z0)
solve!(solver)
max_violation(solver)
visualize!(vis, solver)
plot(states(solver), 7:9)
conSet = get_constraints(solver)
conSet.constraints

opts_al.verbose = false
@btime begin
    initial_trajectory!($solver, $Z0)
    solve!($solver)
end

al = solver.solver_al
@btime begin
    initial_trajectory!($al, $Z0)
    solve!($al)
end

initial_trajectory!(solver,Z0)
al = solver.solver_al
initialize!(al)
TO.set_tolerances!(al, al.solver_uncon, 1)

TO.step!(al)
max_violation(al)
visualize!(vis, al)
