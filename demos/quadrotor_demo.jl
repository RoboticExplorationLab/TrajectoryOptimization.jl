using TrajectoryOptimization
using Plots
using MeshCatMechanisms
using MeshCat

urdf_folder = joinpath(Pkg.dir("TrajectoryOptimization"), "dynamics/urdf")
urdf = joinpath(urdf_folder, "quadrotor_fla.urdf")

quadrotor = parse_urdf(Float64,urdf)
quad = root_body(quadrotor)
T = Float64
inertia1 = SpatialInertia(CartesianFrame3D("upper_link"))
body1 = RigidBody(inertia1)
MechanismState(quad)

add_body_fixed_frame!(quadrotor,)
default_frame(quadrotor)
const state = MechanismState(quadrotor)
vis = Visualizer()
open(vis)
inspect(state)
mvis = MechanismVisualizer(quadrotor,URDFVisuals(urdf),vis)
set_configuration!(mvis, [0.0, 0.0,0.0])
# open(vis)
(quadrotor)
global_coordinates(state)
#
non_tree_joints(state)
#
# set_configuration!(state, [1.0, 1.0, 1.0])
# set_configuration!(vis, configuration(state))
#
