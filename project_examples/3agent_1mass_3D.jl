model = Dynamics.model_a3_m1
nb,mb = 6,9
na1,ma1 = 6,6
na2,ma2 = 6,6
na3,ma3 = 6,6
N = nb+na1+na2+na3
M = mb+ma1+ma2+ma3

bodies = (:m,:a1,:a2,:a3)
ns = (nb,na1,na2,na3)
ms = (mb,ma1,ma2,ma3)
tf = 1.0
scaling = 1.0
_shift = [10.0;0.0;0.0]

z0 = scaling*[0.;0.;0]
ż0 = [0.;0.;0.]
y10 = scaling*[sqrt(8/9);0.;4/3]
ẏ10 = [0.;0.;0.]
y20 = scaling*[-sqrt(2/9);sqrt(2/3);4/3]
ẏ20 = [0.;0.;0.]
y30 = scaling*[-sqrt(2/9);-sqrt(2/3);4/3]
ẏ30 = [0.;0.;0.]
x0 = [z0;ż0;y10;ẏ10;y20;ẏ20;y30;ẏ30]

norm(z0-y10)
norm(z0-y20)
norm(z0-y30)

zf = z0 + _shift
żf = [0.;0.;0.]
y1f = y10 + _shift
ẏ1f = [0.;0.;0.]
y2f = y20 + _shift
ẏ2f = [0.;0.;0.]
y3f = y30 + _shift
ẏ3f = [0.;0.;0.]
xf = [zf;żf;y1f;ẏ1f;y2f;ẏ2f;y3f;ẏ3f]

d = norm(xf[1:3]-xf[7:9])

norm(zf-y1f)
norm(zf-y2f)
norm(zf-y3f)


Q1 = Diagonal(0.01I,nb)
R1 = Diagonal(0.000001I,mb)
Qf1 = Diagonal(1000.0I,nb)
Q2 = Diagonal(0.01I,na1)
R2 = Diagonal(0.0001I,ma1)
Qf2 = Diagonal(1000.0I,na1)
Q3 = Diagonal(0.01I,na2)
R3 = Diagonal(0.0001I,ma2)
Qf3 = Diagonal(1000.0I,na2)
Q4 = Diagonal(0.01I,na3)
R4 = Diagonal(0.0001I,ma3)
Qf4 = Diagonal(1000.0I,na3)



function cE(c,x::AbstractArray,u)
    c[1] = norm(x[7:9] - x[1:3])^2 - d^2
    c[2] = norm(x[13:15] - x[1:3])^2 -d^2
    c[3] = norm(x[19:21] - x[1:3])^2 -d^2
    c[4] = u[1] + u[13]
    c[5] = u[2] + u[14]
    c[6] = u[3] + u[15]
    c[7] = u[4] + u[19]
    c[8] = u[5] + u[20]
    c[9] = u[6] + u[21]
    c[10] = u[7] + u[25]
    c[11] = u[8] + u[26]
    c[12] = u[9] + u[27]
end

function cE(c,x)
    c[1] = norm(x[7:9] - x[1:3])^2 - d^2
    c[2] = norm(x[13:15] - x[1:3])^2 -d^2
    c[3] = norm(x[19:21] - x[1:3])^2 -d^2
end

function ∇cE(cx,cu,x,u)
    z = x[1:3]
    y1 = x[7:9]
    y2 = x[13:15]
    y3 = x[19:21]
    cx[1,1:3] = 2(z - y1)
    cx[1,7:9] = 2(y1 - z)
    cx[2,1:3] = 2(z - y2)
    cx[2,13:15] = 2(y2 - z)
    cx[3,1:3] = 2(z - y3)
    cx[3,19:21] = 2(y3 - z)

    cu[4,1] = 1
    cu[4,13] = 1
    cu[5,2] = 1
    cu[5,14] = 1
    cu[6,3] = 1
    cu[6,15] = 1
    cu[7,4] = 1
    cu[7,19] = 1
    cu[8,5] = 1
    cu[8,20] = 1
    cu[9,6] = 1
    cu[9,21] = 1
    cu[10,7] = 1
    cu[10,25] = 1
    cu[11,8] = 1
    cu[11,26] = 1
    cu[12,9] = 1
    cu[12,27] = 1
end

function ∇cE(cx,x)
    z = x[1:3]
    y1 = x[7:9]
    y2 = x[13:15]
    y3 = x[19:21]
    cx[1,1:3] = 2(z - y1)
    cx[1,7:9] = 2(y1 - z)
    cx[2,1:3] = 2(z - y2)
    cx[2,13:15] = 2(y2 - z)
    cx[3,1:3] = 2(z - y3)
    cx[3,19:21] = 2(y3 - z)
end

cost1 = LQRCost(Q1,R1,Qf1,[zf;żf])
cost2 = LQRCost(Q2,R2,Qf2,[y1f;ẏ1f])
cost3 = LQRCost(Q3,R3,Qf3,[y2f;ẏ2f])
cost4 = LQRCost(Q4,R4,Qf4,[y3f;ẏ3f])

costs = NamedTuple{bodies}((cost1,cost2,cost3,cost4))
part_x = create_partition((nb,na1,na2,na3),bodies)
part_u = create_partition((mb,ma1,ma2,ma3),bodies)

acost = ADMMCost(costs,cE,∇cE,4,[:a1],N,M,part_x,part_u)
obj = UnconstrainedObjective(acost,tf,x0,xf)
obj = ConstrainedObjective(obj,cE=cE,cE_N=cE,∇cE=∇cE,use_xf_equality_constraint=false)
p = obj.p
p_N = obj.p_N

solver = Solver(model,obj,integration=:none,dt=0.1)
solver.opts.cost_tolerance = 1e-5
solver.opts.cost_tolerance_intermediate = 1e-4
solver.opts.constraint_tolerance = 1e-4
solver.opts.penalty_scaling = 2.0
res = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
U0 = zeros(model.m,solver.N-1)
J = admm_solve(solver,res,U0)

admm_plot3(res)

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
sphere_medium = HyperSphere(Point3f0(0), convert(Float32,r_quad))

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
    settransform!(vis["agent1"], Translation(Y1[1:3,i]...))
    settransform!(vis["agent2"], Translation(Y2[1:3,i]...))
    settransform!(vis["agent3"], Translation(Y3[1:3,i]...))
    settransform!(vis["mass1"], Translation(Z[1:3,i]...))

    sleep(solver.dt)
end
