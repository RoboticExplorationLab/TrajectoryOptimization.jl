using MeshCatMechanisms
using RigidBodyDynamics
using MeshCat
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

function animate_trajectory(vis, X::Trajectory, dt=0.1)
    for x in X
        set_configuration!(vis, x[1:7])
        sleep(dt)
    end
end


green_ = MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0))
red_ = MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0))
body_collision = MeshPhongMaterial(color=RGBA(1, 0, 0, 0.5))
blue_ = MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0))
orange_ = MeshPhongMaterial(color=RGBA(233/255, 164/255, 16/255, 1.0))
black_ = MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0))
black_transparent = MeshPhongMaterial(color=RGBA(0, 0, 0, 0.1))
