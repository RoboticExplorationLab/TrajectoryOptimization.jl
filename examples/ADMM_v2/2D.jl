using ForwardDiff, LinearAlgebra, Plots

na = 2
dt = 0.1
n_batch = 8
m_batch = 6
N = 21

function dynamics!(ẋ,x,u)
    ẋ[1:2] = x[3:4]
    ẋ[3:4] = u[1:2]
    ẋ[4] -= 9.81 # gravity
end

function batch_dynamics!(ẋ,x,u)
    u_control = u[1:2]
    u_slack_lift = u[3:4]
    u_slack_load = u[5:6]

    dynamics!(view(ẋ,1:4),x[1:4],u_control + u_slack_lift)
    dynamics!(view(ẋ,5:8),x[5:8],u_slack_load)

    return nothing
end

model = Model(batch_dynamics!,n_batch,m_batch)
model_d = midpoint(model,dt)

x0 = [0.;1;0;0;0;0;0;0]
xf = [10.;1;0;0;10;0;0;0]

d = 1.

Q = Diagonal(ones(n_batch))
R = 0.1*Diagonal(ones(m_batch))
Qf = 100*Diagonal(ones(n_batch))
u0 = [0; 2*9.81;0.;-9.81;0.;9.81]

obj = LQRObjective(Q,R,Qf,xf,N,u0)

function distance_constraint(c,x,u=zeros(m_batch))
    c[1] = norm(x[1:2] - x[5:6])^2 - d^2
    return nothing
end

function direction_constraint(c,x,u)
    Δx = x[5:6] - x[1:2]
    Is = Diagonal(ones(2))
    c[1:2] = (Δx'*Δx*Is - Δx*Δx')*u[3:4]
    c[3:4] = (Δx'*Δx*Is - Δx*Δx')*u[5:6]
    return nothing
end

function force_constraint(c,x,u)
    c[1:2] = u[3:4] + u[5:6]
    return nothing
end

u_l = -Inf*ones(m_batch)
u_u = Inf*ones(m_batch)
u_l[6] = 0.

bnd = BoundConstraint(n_batch,m_batch,u_min=u_l)

dist_con = Constraint{Equality}(distance_constraint,n_batch,m_batch,1,:distance)
dir_con = Constraint{Equality}(direction_constraint,n_batch,m_batch,4,:direction)
for_con = Constraint{Equality}(force_constraint,n_batch,m_batch,2,:force)

goal = goal_constraint(xf)

con = Constraints(N)

for k = 1:N-1
    con[k] += dist_con + dir_con + for_con + bnd
end
con[N] += dist_con + goal
prob = Problem(model_d,obj,dt=dt,N=N,constraints=con,xf=xf,x0=x0)

u0 = [0; 2*9.81;0.;-9.81;0.;9.81]

U0 = [u0 for k = 1:N-1]
initial_controls!(prob,U0)

rollout!(prob)

plot(prob.X,1:2)
plot(prob.X,5:6)

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

plot(prob.X,1:2)
plot(prob.X,5:6)

kk = 3

Δx = prob.X[kk][5:6] - prob.X[kk][1:2]
Δx/norm(Δx)

uu = prob.U[kk][3:4]
ul = prob.U[kk][5:6]

uu/norm(uu)
ul/norm(ul)

plot(prob.U,5:6)

include(joinpath(pwd(),"examples/ADMM/visualization.jl"))

function visualize(vis,prob)

    # camera angle
    # settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))

    # intialize system
    i = 1
    setobject!(vis["lift$i"],HyperSphere(Point3f0(0), convert(Float32,0.05)) ,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))
    cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d),convert(Float32,0.01))
    setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))

    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,0.05)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

    anim = MeshCat.Animation(convert(Int,floor(1.0/prob.dt)))
    for k = 1:prob.N
        MeshCat.atframe(anim,vis,k) do frame
            # cables
            x_load = prob.X[k][5:6]
            x_lift = prob.X[k][1:2]
            settransform!(frame["cable"]["$i"], cable_transform([x_lift[1],0,x_lift[2]],[x_load[1],0,x_load[2]]))
            settransform!(frame["lift$i"], Translation([x_lift[1],0,x_lift[2]]...))

            settransform!(frame["load"], Translation([x_load[1],0,x_load[2]]...))
        end
    end
    MeshCat.setanimation!(vis,anim)
end

vis = Visualizer()
open(vis)
visualize(vis,prob)
