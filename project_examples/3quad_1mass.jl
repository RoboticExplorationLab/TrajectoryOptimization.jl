model = Dynamics.model_3quad_1mass
nb,mb = 6,9
na1,ma1 = 13,7
na2,ma2 = 13,7
na3,ma3 = 13,7

N = nb+na1+na2+na3
M = mb+ma1+ma2+ma3

bodies = (:m,:a1,:a2,:a3)
ns = (nb,na1,na2,na3)
ms = (mb,ma1,ma2,ma3)
tf = 2.25
scaling = 1.0
_shift = [10.0;0.0;0.0]

z0 = scaling*[0.;0.;0]
ż0 = [0.;0.;0.]
y10 = zeros(13)
y10[1:3] = scaling*[sqrt(8/9);0.;4/3]
y10[4] = 1.0
y20 = zeros(13)
y20[1:3] = scaling*[-sqrt(2/9);sqrt(2/3);4/3]
y20[4] = 1.0
y30 = zeros(13)
y30[1:3] = scaling*[-sqrt(2/9);-sqrt(2/3);4/3]
y30[4] = 1.0
x0 = [z0;ż0;y10;y20;y30]

norm(z0-y10[1:3])
norm(z0-y20[1:3])
d = norm(z0-y30[1:3])

zf = z0 + _shift
żf = [0.;0.;0.]
y1f = copy(y10)
y1f[1:3] += _shift
y2f = copy(y20)
y2f[1:3] += _shift
y3f = copy(y30)
y3f[1:3] += _shift
xf = [zf;żf;y1f;y2f;y3f]

Q1 = Diagonal(0.01I,nb)
R1 = Diagonal(0.00001I,mb)
Qf1 = Diagonal(10000.0I,nb)
Q2 = Diagonal(0.01I,na1)
R2 = Diagonal(0.0001I,ma1)
Qf2 = Diagonal(1000.0I,na1)
Q3 = Diagonal(0.01I,na2)
R3 = Diagonal(0.0001I,ma2)
Qf3 = Diagonal(1000.0I,na2)
Q4 = Diagonal(0.01I,na3)
R4 = Diagonal(0.0001I,ma3)
Qf4 = Diagonal(1000.0I,na3)

# Q1 = 0.0001*Matrix(I,nb,nb)
# R1 = 0.00001*Matrix(I,mb,mb)
# Qf1 = 10000.0*Matrix(I,nb,nb)
#
# Q2 = (1e-1)*Matrix(I,na1,na1)
# Q2[4,4] = 1.0; Q2[5,5] = 1.0; Q2[6,6] = 1.0; Q2[7,7] = 1.0
# R2 = (1.0)*Matrix(I,ma1,ma1)
# Qf2 = (1000.0)*Matrix(I,na1,na1)
#
# Q3 = (1e-1)*Matrix(I,na2,na2)
# Q3[4,4] = 1.0; Q3[5,5] = 1.0; Q3[6,6] = 1.0; Q3[7,7] = 1.0
# R3 = (1.0)*Matrix(I,ma2,ma2)
# Qf3 = (1000.0)*Matrix(I,na2,na2)
#
# Q4 = (1e-1)*Matrix(I,na3,na3)
# Q4[4,4] = 1.0; Q4[5,5] = 1.0; Q4[6,6] = 1.0; Q4[7,7] = 1.0
# R4 = (1.0)*Matrix(I,ma3,ma3)
# Qf4 = (1000.0)*Matrix(I,na3,na3)

# Q2 = Diagonal(0.0001I,na1)
# R2 = Diagonal(0.001I,ma1)
# Qf2 = Diagonal(1000.0I,na1)
# Q3 = Diagonal(0.0001I,na2)
# R3 = Diagonal(0.001I,ma2)
# Qf3 = Diagonal(1000.0I,na2)
# Q4 = Diagonal(0.0001I,na3)
# R4 = Diagonal(0.001I,ma3)
# Qf4 = Diagonal(1000.0I,na3)

function cE(c,x::AbstractArray,u)
    c[1] = norm(x[7:9] - x[1:3])^2 - d^2
    c[2] = norm(x[20:22] - x[1:3])^2 -d^2
    c[3] = norm(x[33:35] - x[1:3])^2 -d^2
    c[4] = u[1] + u[14]
    c[5] = u[2] + u[15]
    c[6] = u[3] + u[16]
    c[7] = u[4] + u[21]
    c[8] = u[5] + u[22]
    c[9] = u[6] + u[23]
    c[10] = u[7] + u[28]
    c[11] = u[8] + u[29]
    c[12] = u[9] + u[30]
end

function cE(c,x)
    c[1] = norm(x[7:9] - x[1:3])^2 - d^2
    c[2] = norm(x[20:22] - x[1:3])^2 -d^2
    c[3] = norm(x[33:35] - x[1:3])^2 -d^2
    # c[4:48] = x - xf
end

function ∇cE(cx,cu,x,u)
    z = x[1:3]
    y1 = x[7:9]
    y2 = x[20:22]
    y3 = x[33:35]
    cx[1,1:3] = 2(z - y1)
    cx[1,7:9] = 2(y1 - z)
    cx[2,1:3] = 2(z - y2)
    cx[2,20:22] = 2(y2 - z)
    cx[3,1:3] = 2(z - y3)
    cx[3,33:35] = 2(y3 - z)

    cu[4,1] = 1
    cu[4,14] = 1
    cu[5,2] = 1
    cu[5,15] = 1
    cu[6,3] = 1
    cu[6,16] = 1
    cu[7,4] = 1
    cu[7,21] = 1
    cu[8,5] = 1
    cu[8,22] = 1
    cu[9,6] = 1
    cu[9,23] = 1
    cu[10,7] = 1
    cu[10,28] = 1
    cu[11,8] = 1
    cu[11,29] = 1
    cu[12,9] = 1
    cu[12,30] = 1
end

function ∇cE(cx,x)
    z = x[1:3]
    y1 = x[7:9]
    y2 = x[20:22]
    y3 = x[33:35]
    cx[1,1:3] = 2(z - y1)
    cx[1,7:9] = 2(y1 - z)
    cx[2,1:3] = 2(z - y2)
    cx[2,20:22] = 2(y2 - z)
    cx[3,1:3] = 2(z - y3)
    cx[3,33:35] = 2(y3 - z)
    # cx[4:48,1:45] = 1.0*Matrix(I,45,45)
end

cost1 = LQRCost(Q1,R1,Qf1,[zf;żf])
cost2 = LQRCost(Q2,R2,Qf2,y1f)
cost3 = LQRCost(Q3,R3,Qf3,y2f)
cost4 = LQRCost(Q4,R4,Qf4,y3f)
costs = NamedTuple{bodies}((cost1,cost2,cost3,cost4))
part_x = create_partition((nb,na1,na2,na3),bodies)
part_u = create_partition((mb,ma1,ma2,ma3),bodies)

acost = ADMMCost(costs,cE,∇cE,4,[:a1],N,M,part_x,part_u)
obj = UnconstrainedObjective(acost,tf,x0,xf)
obj = ConstrainedObjective(obj,cE=cE,cE_N=cE,∇cE=∇cE,use_xf_equality_constraint=false)
p = obj.p
p_N = obj.p_N
solver = Solver(model,obj,integration=:rk3,dt=0.1)
solver.opts.cost_tolerance = 1e-5
solver.opts.cost_tolerance_intermediate = 1e-5
solver.opts.constraint_tolerance = 1e-4
solver.opts.penalty_scaling = 2.0
res = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
U0 = rand(model.m,solver.N-1)
U0[10:13,:] .= 0.5*9.81/4.0
U0[17:20,:] .= 0.5*9.81/4.0
U0[24:27,:] .= 0.5*9.81/4.0
X = rollout(solver,U0)
@time J = admm_solve(solver,res,U0)

# 3D visualization
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO

vis = Visualizer()
open(vis)

# Import quadrotor obj file
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics","urdf")
obj_quad = joinpath(urdf_folder, "quadrotor_base.obj")

# color options
green_ = MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0))
green_transparent = MeshPhongMaterial(color=RGBA(0, 1, 0, 0.1))
red_ = MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0))
red_transparent = MeshPhongMaterial(color=RGBA(1, 0, 0, 0.1))
blue_ = MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0))
blue_transparent = MeshPhongMaterial(color=RGBA(0, 0, 1, 0.1))
blue_semi = MeshPhongMaterial(color=RGBA(0, 0, 1, 0.5))
yellow_ = MeshPhongMaterial(color=RGBA(1, 1, 0, 1.0))
yellow_transparent = MeshPhongMaterial(color=RGBA(1, 1, 0, 0.75))

orange_ = MeshPhongMaterial(color=RGBA(233/255, 164/255, 16/255, 1.0))
orange_transparent = MeshPhongMaterial(color=RGBA(233/255, 164/255, 16/255, 0.1))
black_ = MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0))
black_transparent = MeshPhongMaterial(color=RGBA(0, 0, 0, 0.1))
black_semi = MeshPhongMaterial(color=RGBA(0, 0, 0, 0.5))

# geometries
quad_scaling = 0.1
robot_obj = FileIO.load(obj_quad)
robot_obj.vertices .= robot_obj.vertices .* quad_scaling

sphere_small = HyperSphere(Point3f0(0), convert(Float32,0.15)) # trajectory points
sphere_medium = HyperSphere(Point3f0(0), convert(Float32,1.0))


agent1 = vis["agent1"]
agent2 = vis["agent2"]
agent3 = vis["agent3"]
mass1 = vis["mass1"]

Z = to_array(res.X)[part_x.m,:]
Y1 = to_array(res.X)[part_x.a1,:]
Y2 = to_array(res.X)[part_x.a2,:]
Y3 = to_array(res.X)[part_x.a3,:]

# Set camera location
settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))
setobject!(vis["agent1"],robot_obj,black_)
setobject!(vis["agent2"],robot_obj,black_)
setobject!(vis["agent3"],robot_obj,black_)
setobject!(vis["mass1"],sphere_small,green_)

for i = 1:solver.N
    # cables
    geom = Cylinder(Point3f0([Z[1,i],Z[2,i],Z[3,i]]),Point3f0([Y1[1,i],Y1[2,i],Y1[3,i]]),convert(Float32,0.01))
    setobject!(vis["cable"]["1"],geom,red_)
    geom = Cylinder(Point3f0([Z[1,i],Z[2,i],Z[3,i]]),Point3f0([Y2[1,i],Y2[2,i],Y2[3,i]]),convert(Float32,0.01))
    setobject!(vis["cable"]["2"],geom,red_)
    geom = Cylinder(Point3f0([Z[1,i],Z[2,i],Z[3,i]]),Point3f0([Y3[1,i],Y3[2,i],Y3[3,i]]),convert(Float32,0.01))
    setobject!(vis["cable"]["3"],geom,red_)

    # agents + load
    settransform!(vis["agent1"], compose(Translation(Y1[1:3,i]...),LinearMap(Quat(Y1[4:7,i]...))))
    settransform!(vis["agent2"], compose(Translation(Y2[1:3,i]...),LinearMap(Quat(Y2[4:7,i]...))))
    settransform!(vis["agent3"], compose(Translation(Y3[1:3,i]...),LinearMap(Quat(Y3[4:7,i]...))))
    settransform!(vis["mass1"], Translation(Z[1:3,i]...))

    sleep(solver.dt)

end
