## Acrobot
#TODO test
urdf_folder = joinpath(Pkg.dir("TrajectoryOptimization"), "dynamics/urdf")
urdf_doublependulum = joinpath(urdf_folder, "doublependulum.urdf")

model = Model(urdf_doublependulum,[0.;1.]) # underactuated, only control for second "elbow" joint

# initial and goal states
x0 = [0.;0.;0.;0.]
xf = [pi;0.;0.;0.]

# costs
Q = 1e-4*eye(model.n)
Qf = 250.0*eye(model.n)
R = 1e-4*eye(model.m)

# simulation
tf = 5.0
dt = 0.01

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

acrobot = [model, obj_uncon]
