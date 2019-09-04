using ForwardDiff, LinearAlgebra, Plots, StaticArrays
const TO = TrajectoryOptimization
include(joinpath(dirname(@__FILE__),"../ADMM/visualization.jl"))
include(joinpath(dirname(@__FILE__),"problem.jl"))

function visualize(vis,prob,num_lift=3)

    # camera angle
    # settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))
    _cyl = door_obstacles()
    addcylinders!(vis, _cyl, 2.1)
    x0 = prob.x0
    d = norm(x0[1:3] - x0[num_lift*13 .+ (1:3)])

    # intialize system
    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.085
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling
    for i = 1:num_lift
        setobject!(vis["lift$i"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))
        cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d),convert(Float32,0.01))
        setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
    end
    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,0.2)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

    anim = MeshCat.Animation(convert(Int,floor(1.0/prob.dt)))
    for k = 1:prob.N
        MeshCat.atframe(anim,vis,k) do frame
            # cables
            x_load = prob.X[k][3*13 .+ (1:3)]
            for i = 1:num_lift
                x_lift = prob.X[k][(i-1)*13 .+ (1:3)]
                settransform!(frame["cable"]["$i"], cable_transform(x_lift,x_load))
                settransform!(frame["lift$i"], Translation(x_lift...))
            end
            settransform!(frame["load"], Translation(x_load...))
        end
    end
    MeshCat.setanimation!(vis,anim)
end

# Solver options
verbose=false

opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=250)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-4,
    iterations=20,
    penalty_scaling=10.0,
    penalty_initial=1.0e-3)


# Create Problem
prob = gen_prob_batch(quad_params, load_params, batch=true)

# @btime solve($prob,$opts_al)
@time solve!(prob,opts_al)
max_violation(prob)
TO.findmax_violation(prob)

vis = Visualizer()
open(vis)
visualize(vis,prob)

#=
Notes:
Fastest solve with midpoint cost = 10.0
Smoothest solution with midpoint cost = 1.0
=#
