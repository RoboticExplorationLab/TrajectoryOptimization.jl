Diagonal(I,n)## Acrobot
#TODO test
import TrajectoryOptimization
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf_doublependulum = joinpath(urdf_folder, "doublependulum.urdf")

model = Model(urdf_doublependulum,[0.;1.]) # underactuated, only control for second "elbow" joint
n,m = model.n, model.m

# initial and goal states
x0 = [0.;0.;0.;0.]
xf = [pi;0.;0.;0.]

# costs
Q = 1e-4*Diagonal(I,n)
Qf = 250.0*Diagonal(I,n)
R = 1e-4*Diagonal(I,m)

# simulation
tf = 5.0
dt = 0.01

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

acrobot = [model, obj_uncon]
