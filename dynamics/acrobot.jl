import TrajectoryOptimization
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf_doublependulum = joinpath(urdf_folder, "doublependulum.urdf")

acrobot_model = Model(urdf_doublependulum,[0.;1.]) # underactuated, only control for second "elbow" joint
