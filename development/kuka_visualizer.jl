using MeshCatMechanisms
using RigidBodyDynamics
# using Blink
# AtomShell.isinstalled() || AtomShell.install()
# traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
# urdf_folder = joinpath(traj_folder, "dynamics/urdf")
# urdf_kuka_orig = joinpath(urdf_folder, "kuka_iiwa.urdf")
# urdf_kuka = joinpath(urdf_folder, "kuka.urdf")
#
# mech = parse_urdf(urdf_kuka)
# visuals = URDFVisuals(urdf_kuka)
# vis = MechanismVisualizer(mech, visuals)
# open(vis)
# set_configuration!(vis, rand(6))
#
# IJuliaCell(vis)


function animate_trajectory(vis, X::AbstractMatrix{Float64}, dt=0.1)
    animate_trajectory(vis, TrajectoryOptimization.to_dvecs(X))
end

function animate_trajectory(vis, X::Vector{Vector{Float64}}, dt=0.1)
    for x in X
        set_configuration!(vis, x[1:7])
        sleep(dt)
    end
end
