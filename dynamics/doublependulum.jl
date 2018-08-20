using LinearAlgebra

## Double Pendulum
# TODO test
import TrajectoryOptimization
urdf_folder = joinpath(dirname(pathof(TrajectoryOptimization)), "..", "dynamics/urdf")
urdf_doublependulum = joinpath(urdf_folder, "doublependulum.urdf")
#isfile(urdf_doublependulum)

model = Model(urdf_doublependulum)
n = model.n; # dimensions of system
m = model.m; # dimensions of control

# initial and goal states
x0 = [0.;0.;0.;0.]
xf = [pi;0.;0.;0.]

# costs
Q = 0.0001*Diagonal{Float64}(I, model.n)
Qf = 250.0*Diagonal{Float64}(I, model.n)
R = 0.0001*Diagonal{Float64}(I, model.m)

# simulation
tf = 5.0

obj = UnconstrainedObjective(Q,R,Qf,tf,x0,xf)
doublependulum = [model,obj]
