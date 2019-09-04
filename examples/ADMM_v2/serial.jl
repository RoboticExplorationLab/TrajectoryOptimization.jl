using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using LinearAlgebra
using ForwardDiff
const TO = TrajectoryOptimization

include("problem.jl")
include("methods.jl")

# Solver options
verbose=true

opts_ilqr = iLQRSolverOptions(verbose=true,
      iterations=500)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=0.001,
    cost_tolerance_intermediate=1.0e-4,
    iterations=30,
    penalty_scaling=2.0,
    penalty_initial=10.0)

num_lift = 3
prob_load = gen_prob(:load)
prob_lift = [gen_prob(i) for i = 1:num_lift]
@time plift_al, pload_al, slift_al, sload_al = solve_admm_1slack(prob_lift,prob_load,:parallel,opts_al)
# @time plift_al, pload_al, slift_al, sload_al = solve_admm(prob_lift,prob_load,n_slack,:sequential,opts_al)

max_violation(slift_al[1])
max_violation(slift_al[2])
max_violation(slift_al[3])
max_violation(sload_al)

# Visualize
include(joinpath(pwd(),"examples/ADMM/visualization.jl"))

vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, [[pload_al]; plift_al])

idx = [(1:3)...,(8:10)...]
plot(plift_al[1].U,label="")
plot(plift_al[1].X,8:10)
plot(plift_al[1].X,1:3)



output_traj(plift_al[1],idx,joinpath(pwd(),"examples/ADMM/traj0.txt"))
output_traj(plift_al[2],idx,joinpath(pwd(),"examples/ADMM/traj1.txt"))
output_traj(plift_al[3],idx,joinpath(pwd(),"examples/ADMM/traj2.txt"))


# for i = 1:num_lift
#     rollout!(prob_lift[i])
# end
# rollout!(prob_load)
#
# vis = Visualizer()
# open(vis)
# visualize_quadrotor_lift_system(vis, [[prob_load]; prob_lift],_cyl)

# plot(prob.X,1:3)
# plot(prob.X,13 .+ (1:3))
#
kk = 3

Δx = pload_al.X[kk][(1:3)] - plift_al[1].X[kk][1:3]
Δx/norm(Δx)

uu = plift_al[1].U[kk][5]
ul = pload_al.U[kk][1]

uu/norm(uu)
ul/norm(ul)

norm(uu)
norm(ul)

plot(plift_al[1].U,1:4)
plot(plift_al[1].U,5:5)
plot(pload_al.U,1:1)

plot(plift_al[2].U,5:5)
plot(pload_al.U,2:2)

plot(plift_al[3].U,5:5)
plot(pload_al.U,3:3)

#
# plot(plift_al[1].U,1:4)
# plot(plift_al[1].U,4 .+ (1:3))
# plot(pload_al.U,(1:3))
#
# plot(prob.U,(7 + 4) .+ (1:3))
# plot(prob.U,(7*3 + 3) .+ (1:3))
