## Double Pendulum
# TODO test
import TrajectoryOptimization
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf_doublependulum = joinpath(urdf_folder, "doublependulum.urdf")
#isfile(urdf_doublependulum)

model = Model(urdf_doublependulum)
n = model.n; # dimensions of system
m = model.m; # dimensions of control

# initial and goal states
x0 = [0.;0.;0.;0.]
xf = [pi;0.;0.;0.]

# costs
Q = 0.0001*Diagonal(I,n)
Qf = 250.0*Diagonal(I,n)
R = 0.0001*Diagonal(I,m)

# simulation
tf = 5.0

obj = LQRObjective(Q,R,Qf,tf,x0,xf)
doublependulum = [model,obj]
