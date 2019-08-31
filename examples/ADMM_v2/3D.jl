using ForwardDiff, LinearAlgebra, Plots, StaticArrays

na = 2
dt = 0.1
n_batch = 13 + 6
m_batch = 4 + 3 + 3
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
load_mass = 0.35

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
    lift_control = u[1:7]
    u_slack_lift = u[5:7]
    u_slack_load = u[8:10]

    lift_dynamics!(view(ẋ,1:13),x[1:13],lift_control,lift_params)
    load_dynamics!(view(ẋ,14:19),x[14:19],u_slack_load/load_mass)

    return nothing
end

model = Model(batch_dynamics!,n_batch,m_batch)
model_d = midpoint(model,dt)

# model_d.f(rand(n_batch),rand(n_batch),rand(m_batch),0.1)

x0 = zeros(n_batch)
x0[3] = 1.
x0[4] = 1.
xf = copy(x0)
xf[1] = 10.
xf[14] = 10.

d = 1.

Q = Diagonal(ones(n_batch))
R = 0.1*Diagonal(ones(m_batch))
Qf = 100*Diagonal(ones(n_batch))

thrust = 9.81*(lift_params.m + load_mass)/4
u0 = [thrust;thrust;thrust;thrust;0.;0.;-9.81*load_mass;0.;0.;9.81*load_mass]

obj = LQRObjective(Q,R,Qf,xf,N,u0)

function distance_constraint(c,x,u=zeros(m_batch))
    c[1] = norm(x[1:3] - x[13 .+ (1:3)])^2 - d^2
    return nothing
end

function direction_constraint(c,x,u)
    Δx = x[13 .+ (1:3)] - x[1:3]
    Is = Diagonal(ones(3))
    c[1:3] = (Δx'*Δx*Is - Δx*Δx')*u[4 .+ (1:3)]
    c[4:6] = (Δx'*Δx*Is - Δx*Δx')*u[7 .+ (1:3)]
    return nothing
end

function force_constraint(c,x,u)
    c[1:3] = u[4 .+ (1:3)] + u[7 .+ (1:3)]
    return nothing
end

u_l = -Inf*ones(m_batch)
u_u = Inf*ones(m_batch)
u_l[10] = 0.

bnd = BoundConstraint(n_batch,m_batch,u_min=u_l)

dist_con = Constraint{Equality}(distance_constraint,n_batch,m_batch,1,:distance)
dir_con = Constraint{Equality}(direction_constraint,n_batch,m_batch,6,:direction)
for_con = Constraint{Equality}(force_constraint,n_batch,m_batch,3,:force)

goal = goal_constraint(xf)

con = Constraints(N)

for k = 1:N-1
    con[k] += dist_con + dir_con + for_con + bnd
end
con[N] += dist_con + goal
prob = Problem(model_d,obj,dt=dt,N=N,constraints=con,xf=xf,x0=x0)


U0 = [u0 for k = 1:N-1]
initial_controls!(prob,U0)

rollout!(prob)

plot(prob.X,1:3)
plot(prob.X,13 .+ (1:3))

verbose=true

opts_ilqr = iLQRSolverOptions(verbose=true,
      iterations=250)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-4,
    iterations=10,
    penalty_scaling=2.0,
    penalty_initial=10.)

solve!(prob,opts_al)

plot(prob.X,1:3)
plot(prob.X,13 .+ (1:3))

kk = 3

Δx = prob.X[kk][13 .+ (1:3)] - prob.X[kk][1:3]
Δx/norm(Δx)

uu = prob.U[kk][4 .+ (1:3)]
ul = prob.U[kk][7 .+ (1:3)]

uu/norm(uu)
ul/norm(ul)

plot(prob.U,4 .+ (1:3))
plot(prob.U,7 .+ (1:3))

include(joinpath(pwd(),"examples/ADMM/visualization.jl"))

function visualize(vis,prob)

    # camera angle
    # settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))

    # intialize system
    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.07
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling
    i = 1
    setobject!(vis["lift$i"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))
    cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d),convert(Float32,0.01))
    setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))

    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,0.05)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

    anim = MeshCat.Animation(convert(Int,floor(1.0/prob.dt)))
    for k = 1:prob.N
        MeshCat.atframe(anim,vis,k) do frame
            # cables
            x_load = prob.X[k][13 .+ (1:3)]
            x_lift = prob.X[k][1:3]
            settransform!(frame["cable"]["$i"], cable_transform(x_lift,x_load))
            settransform!(frame["lift$i"], Translation(x_lift...))

            settransform!(frame["load"], Translation(x_load...))
        end
    end
    MeshCat.setanimation!(vis,anim)
end

vis = Visualizer()
open(vis)
visualize(vis,prob)
