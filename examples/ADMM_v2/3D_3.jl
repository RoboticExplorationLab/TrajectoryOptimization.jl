using ForwardDiff, LinearAlgebra, Plots, StaticArrays

na = 3
dt = 0.2
n_batch = na*13 + 6
m_batch = na*(4 + 3) + 3*na
N = 51

function load_dynamics!(ẋ,x,u)
    ẋ[1:3] = x[4:6]
    ẋ[4:6] = u[1:3]
    ẋ[6] -= 9.81 # gravity
end

include(joinpath(pwd(),"dynamics/quaternions.jl"))
lift_params = (m=0.85,
             J=SMatrix{3,3}(Diagonal([0.0023, 0.0023, 0.004])),
             Jinv=SMatrix{3,3}(Diagonal(1.0./[0.0023, 0.0023, 0.004])),
             gravity=SVector(0,0,-9.81),
             motor_dist=0.1750,
             kf=1.0,
             km=0.0245)
n_lift = 13
load_mass = 0.1
n_load = 6

function lift_dynamics!(ẋ,x,u,params)

      q = normalize(Quaternion(view(x,4:7)))
      v = view(x,8:10)
      omega = view(x,11:13)

      # Parameters
      m = params[:m] # mass
      J = params[:J] # inertia matrix
      Jinv = params[:Jinv] # inverted inertia matrix
      g = params[:gravity] # gravity
      L = params[:motor_dist] # distance between motors

      w1 = u[1]
      w2 = u[2]
      w3 = u[3]
      w4 = u[4]

      kf = params[:kf]; # 6.11*10^-8;
      F1 = kf*w1;
      F2 = kf*w2;
      F3 = kf*w3;
      F4 = kf*w4;
      F = @SVector [0., 0., F1+F2+F3+F4] #total rotor force in body frame

      km = params[:km]
      M1 = km*w1;
      M2 = km*w2;
      M3 = km*w3;
      M4 = km*w4;
      tau = @SVector [L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame

      ẋ[1:3] = v # velocity in world frame
      ẋ[4:7] = SVector(0.5*q*Quaternion(zero(x[1]), omega...))
      ẋ[8:10] = g + (1/m)*(q*F + u[5:7]) #acceleration in world frame
      ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
      return tau, omega, J, Jinv
end

function batch_dynamics!(ẋ,x,u)
    lift_control_1 = u[1:7]
    lift_control_2 = u[7 .+ (1:7)]
    lift_control_3 = u[2*7 .+ (1:7)]
    u_slack_load = u[3*7 .+ (1:3)] + u[(3*7 + 3) .+ (1:3)] + u[(3*7 + 2*3) .+ (1:3)]

    lift_dynamics!(view(ẋ,1:13),x[1:13],lift_control_1,lift_params)
    lift_dynamics!(view(ẋ,13 .+ (1:13)),x[13 .+ (1:13)],lift_control_2,lift_params)
    lift_dynamics!(view(ẋ,2*13 .+ (1:13)),x[2*13 .+ (1:13)],lift_control_3,lift_params)

    load_dynamics!(view(ẋ,3*13 .+ (1:6)),x[3*13 .+ (1:6)],u_slack_load/load_mass)

    return nothing
end

model = Model(batch_dynamics!,n_batch,m_batch)
model_d = midpoint(model,dt)

model_d.f(rand(n_batch),rand(n_batch),rand(m_batch),0.1)

goal_dist = 10.0

shift_ = zeros(n_lift)
shift_[1:3] = [0.0;0.0;0.25]
scaling = 1.25
x10 = zeros(n_lift)
x10[4] = 1.
x10[1:3] = scaling*[sqrt(8/9);0.;4/3]
x10 += shift_
x20 = zeros(n_lift)
x20[4] = 1.
x20[1:3] = scaling*[-sqrt(2/9);sqrt(2/3);4/3]
x20 += shift_
x30 = zeros(n_lift)
x30[4] = 1.
x30[1:3] = scaling*[-sqrt(2/9);-sqrt(2/3);4/3]
x30 += shift_
xload0 = zeros(n_load)
xload0[3] = 3/6
xload0[1:3] += shift_[1:3]

xlift0 = [x10,x20,x30]

_shift = zeros(n_lift)
_shift[1:3] = [goal_dist;0.0;0.0]

# goal state
xloadf = zeros(n_load)
xloadf[1:3] = xload0[1:3] + _shift[1:3]
x1f = copy(x10) + _shift
x2f = copy(x20) + _shift
x3f = copy(x30) + _shift

xliftf = [x1f,x2f,x3f]
# xliftf = xlift0
# xloadf = xload0

x0 = vcat(xlift0...,xload0)
xf = vcat(xliftf...,xloadf)
d = norm(x10[1:3]-xload0[1:3])

Q = 1.0e-1*Diagonal(ones(n_batch))
for i in [1,13+1,2*13+1,3*13+1]
      Q[i,i] = 1.0e-3
end
for i in [(3*13 .+ (1:6))...]
      Q[i,i] = 0.
end
r_control = 1.0e-3*ones(4)
r_slack = ones(3)
R = Diagonal([r_control;r_slack;r_control;r_slack;r_control;r_slack;r_slack;r_slack;r_slack])
Qf = 100*Diagonal(ones(n_batch))
for i in [(3*13 .+ (1:6))...]
      Qf[i,i] = 0.
end

# determine static forces
f1 = (x10[1:3] - xload0[1:3])/norm(x10[1:3] - xload0[1:3])
f2 = (x20[1:3] - xload0[1:3])/norm(x20[1:3] - xload0[1:3])
f3 = (x30[1:3] - xload0[1:3])/norm(x30[1:3] - xload0[1:3])
f_mag = hcat(f1, f2, f3)\[0;0;9.81*load_mass]
ff = [f_mag[1]*f1, f_mag[2]*f2, f_mag[3]*f3]

thrust = 9.81*(lift_params.m + load_mass/na)/4
ulift = [[thrust;thrust;thrust;thrust;-ff[i]] for i = 1:na]
ulift_r = [[0.;0.;0.;0.;-ff[i]] for i = 1:na]
uload = vcat(ff...)
u0 = vcat(ulift...,uload)
u0_r = vcat(ulift_r...,uload)

obj = LQRObjective(Q,R,Qf,xf,N,u0_r)

# midpoint desired configuration
Nmid = convert(Int,floor(N/2))+1

ℓ1 = norm(x30[1:3]-x10[1:3])
norm(x10[1:3]-x20[1:3])
norm(x20[1:3]-x30[1:3])

ℓ2 = norm(xload0[1:3]-x10[1:3])
norm(xload0[1:3]-x20[1:3])
norm(xload0[1:3]-x30[1:3])

ℓ3 = 0.

_shift_ = zeros(n_lift)
_shift_[1] = goal_dist/2.

x1m = copy(x10)
x1m += _shift_
x1m[1] += ℓ3
x3m = copy(x1m)
x3m[1] -= ℓ1
x3m[1] -= ℓ3
x3m[2] = -0.01
x2m = copy(x1m)
x2m[1] = goal_dist/2.
x2m[2] = 0.01
x2m[3] = ℓ2 - 0.5*sqrt(4*ℓ2^2 - ℓ1^2 + ℓ3*2) + x20[3]

xliftmid = [x1m,x2m,x3m]

xloadm = [goal_dist/2.; 0; xload0[3];0.;0.;0.]
xm = vcat(xliftmid...,xloadm)
f1m = (x1m[1:3] - xloadm[1:3])/norm(x1m[1:3] - xloadm[1:3])
f2m = (x2m[1:3] - xloadm[1:3])/norm(x2m[1:3] - xloadm[1:3])
f3m = (x3m[1:3] - xloadm[1:3])/norm(x3m[1:3] - xloadm[1:3])
f_magm = hcat(f1m, f2m, f3m)\[0;0;9.81*di_mass_load]
ffm = [f_magm[1]*f1m, f_magm[2]*f2m, f_magm[3]*f3m]

thrustm = 9.81*(quad_params.m + di_mass_load/num_lift)/4
uliftm = [[thrustm;thrustm;thrustm;thrustm;-ffm[i]] for i = 1:num_lift]
uloadm = vcat(ffm...)
um = vcat(uliftm...,uloadm)
Q_mid = copy(Q)

for i in [(1:3)...,(13 .+ (1:3))...,(2*13 .+ (1:3))...,(3*13 .+ (1:3))...]
      Q_mid[i,i] = 1000.
end

cost_mid = LQRCost(Q_mid,R,xm,um)
#
obj.cost[Nmid] = cost_mid

function distance_constraint(c,x,u=zeros(m_batch))
    xload = x[3*13 .+ (1:3)]
    c[1] = norm(x[1:3] - xload)^2 - d^2
    c[2] = norm(x[13 .+ (1:3)] - xload)^2 - d^2
    c[3] = norm(x[2*13 .+ (1:3)] - xload)^2 - d^2

    return nothing
end

function direction_constraint(c,x,u)
    xload = x[3*13 .+ (1:3)]

    Δx1 = xload - x[1:3]
    Δx2 = xload - x[13 .+ (1:3)]
    Δx3 = xload - x[2*13 .+ (1:3)]

    Is = Diagonal(ones(3))
    c[1:3] = (Δx1'*Δx1*Is - Δx1*Δx1')*u[4 .+ (1:3)]
    c[4:6] = (Δx1'*Δx1*Is - Δx1*Δx1')*u[3*7 .+ (1:3)]

    c[7:9] = (Δx2'*Δx2*Is - Δx2*Δx2')*u[(7 + 4) .+ (1:3)]
    c[10:12] = (Δx2'*Δx2*Is - Δx2*Δx2')*u[(3*7 + 3) .+ (1:3)]

    c[13:15] = (Δx3'*Δx3*Is - Δx3*Δx3')*u[(2*7 + 4) .+ (1:3)]
    c[16:18] = (Δx3'*Δx3*Is - Δx3*Δx3')*u[(3*7 + 2*3) .+ (1:3)]
    return nothing
end

function force_constraint(c,x,u)
    c[1:3] = u[4 .+ (1:3)] + u[3*7 .+ (1:3)]
    c[4:6] = u[(7 + 4) .+ (1:3)] + u[(3*7 + 3) .+ (1:3)]
    c[7:9] = u[(2*7 + 4) .+ (1:3)] + u[(3*7 + 2*3) .+ (1:3)]
    return nothing
end

function collision_constraint(c,x,u=zeros(m_batch))
    x1 = x[1:3]
    x2 = x[13 .+ (1:3)]
    x3 = x[2*13 .+ (1:3)]

    c[1] = circle_constraint(x1,x2[1],x2[2],2*r_lift)
    c[2] = circle_constraint(x2,x3[1],x3[2],2*r_lift)
    c[3] = circle_constraint(x3,x1[1],x1[2],2*r_lift)

    return nothing
end

r_cylinder = 0.5
_cyl = []

push!(_cyl,(goal_dist/2.,1.,r_cylinder))
push!(_cyl,(goal_dist/2.,-1.,r_cylinder))

function cI_cylinder(c,x,u)
    c_shift = 1
    n_slack = 3
    for p = 1:length(_cyl)
        n_shift = 0
        for i = 1:num_lift
            idx_pos = (n_shift .+ (1:13))[1:3]
            c[c_shift] = circle_constraint(x[idx_pos],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*r_lift)
            c_shift += 1
            n_shift += 13
        end
        c[c_shift] = circle_constraint(x[3*13 .+ (1:3)],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*r_lift)
        c_shift += 1
    end
end
cyl = Constraint{Inequality}(cI_cylinder,n_batch,m_batch,(num_lift+1)*length(_cyl),:cyl)


u_l = -Inf*ones(m_batch)
u_u = Inf*ones(m_batch)
u_l[1:4] .= 0.
u_l[7 .+ (1:4)] .= 0.
u_l[2*7 .+ (1:4)] .= 0.
u_l[na*7 + 3] = 0.

u_u[1:4] .= 12/4.
u_u[7 .+ (1:4)] .= 12/4.
u_u[2*7 .+ (1:4)] .= 12/4.

bnd = BoundConstraint(n_batch,m_batch,u_min=u_l,u_max=u_u)

dist_con = Constraint{Equality}(distance_constraint,n_batch,m_batch,na,:distance)
dir_con = Constraint{Equality}(direction_constraint,n_batch,m_batch,18,:direction)
for_con = Constraint{Equality}(force_constraint,n_batch,m_batch,9,:force)
col_con = Constraint{Inequality}(collision_constraint,n_batch,m_batch,3,:collision)
goal = goal_constraint(xf)

con = Constraints(N)

for k = 1:N-1
    con[k] += dist_con + dir_con + for_con + bnd + col_con + cyl
end
con[N] += dist_con + goal + col_con + cyl
prob = Problem(model_d,obj,dt=dt,N=N,constraints=con,xf=xf,x0=x0)


U0 = [u0 for k = 1:N-1]
initial_controls!(prob,U0)

rollout!(prob)

plot(prob.X,1:3)
plot(prob.X,13 .+ (1:3))
plot(prob.X,2*13 .+ (1:3))

verbose=true

opts_ilqr = iLQRSolverOptions(verbose=true,
      iterations=250)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-4,
    iterations=20,
    penalty_scaling=10.0,
    penalty_initial=1.0e-3)

@time solve!(prob,opts_al)
max_violation(prob)

plot(prob.X,1:3)
plot(prob.X,13 .+ (1:3))

kk = 3

Δx = prob.X[kk][13*3 .+ (1:3)] - prob.X[kk][1:3]
Δx/norm(Δx)

uu = prob.U[kk][4 .+ (1:3)]
ul = prob.U[kk][3*7 .+ (1:3)]

uu/norm(uu)
ul/norm(ul)

plot(prob.U,1:4)
plot(prob.U,4 .+ (1:3))
plot(prob.U,7*3 .+ (1:3))

plot(prob.U,(7 + 4) .+ (1:3))
plot(prob.U,(7*3 + 3) .+ (1:3))

# plot(prob.U,7 .+ (1:3))

include(joinpath(pwd(),"examples/ADMM/visualization.jl"))

function visualize(vis,prob)

    # camera angle
    # settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))
    addcylinders!(vis, _cyl, 2.1)

    # intialize system
    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.07
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling
    for i = 1:na
        setobject!(vis["lift$i"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))
        cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d),convert(Float32,0.01))
        setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
    end
    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,0.05)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

    anim = MeshCat.Animation(convert(Int,floor(1.0/prob.dt)))
    for k = 1:prob.N
        MeshCat.atframe(anim,vis,k) do frame
            # cables
            x_load = prob.X[k][3*13 .+ (1:3)]
            for i = 1:na
                x_lift = prob.X[k][(i-1)*13 .+ (1:3)]
                settransform!(frame["cable"]["$i"], cable_transform(x_lift,x_load))
                settransform!(frame["lift$i"], Translation(x_lift...))
            end
            settransform!(frame["load"], Translation(x_load...))
        end
    end
    MeshCat.setanimation!(vis,anim)
end

vis = Visualizer()
open(vis)
visualize(vis,prob)
