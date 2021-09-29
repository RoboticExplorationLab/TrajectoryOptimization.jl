using Random
using Logging
using Distributed
using DistributedArrays
using TimerOutputs
using BenchmarkTools

if nworkers() != 15
	addprocs(15,exeflags="--project=$(@__DIR__)")
end
import TrajectoryOptimization: Discrete

using TrajectoryOptimization
const TO = TrajectoryOptimization
@everywhere using TrajectoryOptimization
@everywhere const TO = TrajectoryOptimization
@everywhere using StaticArrays
@everywhere using LinearAlgebra
@everywhere using DistributedArrays
@everywhere using ForwardDiff
@everywhere include(joinpath(dirname(@__FILE__),"problem.jl"))
@everywhere include(joinpath(dirname(@__FILE__),"methods_distributed.jl"))
@everywhere include(joinpath(dirname(@__FILE__),"methods.jl"))
@everywhere include(joinpath(dirname(@__FILE__),"models.jl"))

parallel = true

function get_trajs(; num_lift=3, verbose=false, visualize=true)
    opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=500)

    opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=0.001,
    cost_tolerance_intermediate=1.0e-4,
    iterations=30,
    penalty_scaling=2.0,
    penalty_initial=10.)

    quat = true
    @info "trim"
    prob_lift, prob_load = trim_conditions_dist(num_lift,r0_load,quad_params,load_params,quat,opts_al);
    wait.([@spawnat w reset_control_reference!(prob_lift[:L]) for w in workers()[1:num_lift]])
    @info "auglag"
    sol, solver_lift_al, solver_load_al = solve_admm(prob_lift, prob_load, quad_params,
	    load_params, parallel, opts_al);

    if visualize
        vis = Visualizer()
        open(vis)
        visualize_quadrotor_lift_system(vis,sol,true,num_lift)
        settransform!(vis["/Cameras/default"], compose(Translation(3.5,-5, 5),LinearMap(RotX(0)*RotZ(-pi/2))))
    end
    return sol
end

get_trajs();