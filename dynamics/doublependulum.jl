## Double Pendulum
import TrajectoryOptimization
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf_doublependulum = joinpath(urdf_folder, "doublependulum.urdf")

doublependulum = Model(urdf_doublependulum)
