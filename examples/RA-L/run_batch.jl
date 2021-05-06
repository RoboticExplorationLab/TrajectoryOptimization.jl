using TrajectoryOptimization
const TO = TrajectoryOptimization

include(joinpath(@__DIR__, "visualization.jl"))
include(joinpath(@__DIR__, "problem.jl"))
include(joinpath(@__DIR__, "methods.jl"))

# setting up the problem
function setup(; num_lift=3)
    quat = true
    r0_load = [0,0,0.25]
    scenario = :doorway
    prob = gen_prob(:batch, quad_params, load_params, r0_load, scenario=scenario,num_lift=num_lift,quat=quat);
    return prob
end

function get_trajs(; verbose=true, visualize=true)
    prob = setup()
    opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=50)

    opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-4,
    iterations=50,
    penalty_scaling=10.0,
    penalty_initial=1.0e-3)

    # Step 1:
    @info "trim"
    prob = trim_conditions_batch(num_lift, r0_load, quad_params, load_params, quat, opts_al)

    # Step 2:
    @info "auglag"
    prob, solver = solve(prob, opts_al)

    if visualize
        vis = Visualizer()
        open(vis)
        visualize_batch(vis,prob,true,num_lift)
    end    
    @info "stats" solver.stats[:iterations] max_violation(prob)   
    return prob 
end

get_trajs()
