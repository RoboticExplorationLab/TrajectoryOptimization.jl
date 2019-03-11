model = Dynamics.model_2quad_1mass
nb,mb = 6,3
na1,ma1 = 13,7

N = nb+na1
M = mb+ma1

bodies = (:m,:a1)
ns = (nb,na1)
ms = (mb,ma1)
tf = 1.0
_shift = [10.0;0.0;0.0]

z0 = scaling*[0.;0.;0]
ż0 = [0.;0.;0.]
y10 = zeros(13)
y10[1:3] = [0.;0.;1.]
y10[4:7] = [1.;0.;0.;0.]

x0 = [z0;ż0;y10]
zf = z0 + _shift
żf = [0.;0.;0.]
y1f = copy(y10)
y1f[1:3] += _shift

xf = [zf;żf;y1f]

d = 1

Q1 = Diagonal(0.01I,nb)
R1 = Diagonal(0.00001I,mb)
Qf1 = Diagonal(10000.0I,nb)
Q2 = Diagonal(0.01I,na1)
R2 = Diagonal(0.0001I,ma1)
Qf2 = Diagonal(1000.0I,na1)

function cE(c,x::AbstractArray,u)
    c[1] = norm(x[7:9] - x[1:3])^2 - d^2
    c[2] = u[1] + u[8]
    c[3] = u[2] + u[9]
    c[4] = u[3] + u[10]
end

function cE(c,x)
    c[1] = norm(x[7:9] - x[1:3])^2 - d^2
end

function ∇cE(cx,cu,x,u)
    z = x[1:3]
    y1 = x[7:9]

    cx[1,1:3] = 2(z - y1)
    cx[1,7:9] = 2(y1 - z)

    cu[2,1] = 1
    cu[2,8] = 1
    cu[3,2] = 1
    cu[3,9] = 1
    cu[4,3] = 1
    cu[4,10] = 1
end

function ∇cE(cx,x)
    z = x[1:3]
    y1 = x[7:9]
    cx[1,1:3] = 2(z - y1)
    cx[1,7:9] = 2(y1 - z)
end

cost1 = LQRCost(Q1,R1,Qf1,[zf;żf])
cost2 = LQRCost(Q2,R2,Qf2,y1f)

costs = NamedTuple{bodies}((cost1,cost2))
part_x = create_partition((nb,na1),bodies)
part_u = create_partition((mb,ma1),bodies)

acost = ADMMCost(costs,cE,∇cE,2,[:a1],N,M,part_x,part_u)
obj = UnconstrainedObjective(acost,tf,x0,xf)
obj = ConstrainedObjective(obj,cE=cE,cE_N=cE,∇cE=∇cE,use_xf_equality_constraint=false)
p = obj.p
p_N = obj.p_N

solver = Solver(model,obj,integration=:rk3,dt=0.05)
solver.opts.cost_tolerance = 1e-8
solver.opts.cost_tolerance_intermediate = 1e-8
solver.opts.constraint_tolerance = 1e-6
solver.opts.penalty_scaling = 2.0
res = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
U0 = rand(model.m,solver.N-1)
U0[4:7,:] .= 0.5*9.81/4.0

J = admm_solve(solver,res,U0)

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
mass1 = vis["mass1"]

Z = to_array(res.X)[part_x.m,:]
Y1 = to_array(res.X)[part_x.a1,:]
Z[:,end]
Y1[:,end]
# Set camera location
settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))
setobject!(vis["agent1"],robot_obj,black_)
setobject!(vis["mass1"],sphere_small,green_)

for i = 1:solver.N
    # cables
    geom = Cylinder(Point3f0([Z[1,i],Z[2,i],Z[3,i]]),Point3f0([Y1[1,i],Y1[2,i],Y1[3,i]]),convert(Float32,0.01))
    setobject!(vis["cable"]["1"],geom,red_)

    # agents + load
    settransform!(vis["agent1"], Translation(Y1[1:3,i]...))
    settransform!(vis["mass1"], Translation(Z[1:3,i]...))

    sleep(solver.dt)
end
