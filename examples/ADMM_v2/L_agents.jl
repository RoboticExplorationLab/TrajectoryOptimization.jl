using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using LinearAlgebra
using ForwardDiff
using Combinatorics
using TrajectoryOptimization
const TO = TrajectoryOptimization

include("problem.jl")
include("methods.jl")

# Solver options
verbose=true

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


num_lift = 5
quat = true
r0_load = [0, 0, 0.25]
prob_lift, prob_load = trim_conditions(num_lift,r0_load,quad_params,load_params,quat,opts_al)

# visualize_quadrotor_lift_system(vis, [[prob_load]; prob_lift])


@time plift_al, pload_al, slift_al, sload_al = solve_admm(prob_lift, prob_load, quad_params,
    load_params, :sequential, opts_al, max_iters=10)


include("visualization.jl")
vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, [[pload_al]; plift_al],false)



function trim_conditions(num_lift,r0_load,quad_params,load_params,quat,opts)
    scenario = :hover
    prob_load_trim = gen_prob(:load, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario)
    prob_lift_trim = [gen_prob(i, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario) for i = 1:num_lift]

    lift_trim, load_trim, slift_al, sload_al = solve_admm(prob_lift_trim, prob_load_trim, quad_params,
            load_params, :sequential, opts_al, max_iters=5)

    scenario = :p2p
    prob_load = gen_prob(:load, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario)

    initial_controls!(prob_load,[load_trim.U[end] for k = 1:prob_load.N])
    # rollout!(prob_load)
    prob_lift = [gen_prob(i, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario) for i = 1:num_lift]

    for i = 1:num_lift
        prob_lift[i].x0[4:7] = lift_trim[i].X[end][4:7]
        prob_lift[i].xf[4:7] = lift_trim[i].X[end][4:7]
        initial_controls!(prob_lift[i],[lift_trim[i].U[end] for k = 1:prob_lift[i].N])
        # rollout!(prob_lift[i])
    end

    return prob_lift, prob_load
end
