using iLQR
using RigidBodyDynamics

q = [1;2]
qd = [3;4]
u = [5;2]
dt = 0.1
S = [q;qd;u;dt]
x = [q;qd]


dir = Pkg.dir("DynamicWalking2018")
urdf = joinpath(dir,"notebooks","data","doublependulum.urdf")
doublependulum = parse_urdf(Float64,urdf)

iLQR.Model(doublependulum)
