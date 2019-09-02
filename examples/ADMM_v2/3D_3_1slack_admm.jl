using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using LinearAlgebra
using ForwardDiff
const TO = TrajectoryOptimization

num_lift = 3
N = 101
dt = 0.1

n_lift = 13
m_lift = 5

n_load = 6
m_load = 3

goal_dist = 10.
shift_ = zeros(n_lift)
shift_[1:3] = [0.0;0.0;0.0]
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
xload0[3] = 4/6
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

# midpoint desired configuration
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

include(joinpath(pwd(),"dynamics/quaternions.jl"))

quad_params = (m=0.850,
             J=SMatrix{3,3}(Diagonal([0.0023, 0.0023, 0.004])),
             Jinv=SMatrix{3,3}(Diagonal(1.0./[0.0023, 0.0023, 0.004])),
             gravity=SVector(0,0,-9.81),
             motor_dist=0.175,
             kf=1.0,
             km=0.0245)

function gen_lift_model_initial(agent)
      model = Model[]

      for k = 1:N-1
        function quadrotor_lift_dynamics!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,params)
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
            Δx = xload0[1:3] - xlift0[agent][1:3]
            dir = Δx/norm(Δx)
            ẋ[8:10] = g + (1/m)*(q*F + u[5]*dir) # acceleration in world frame
            ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
            return tau, omega, J, Jinv
        end
        push!(model,midpoint(Model(quadrotor_lift_dynamics!,13,5,quad_params),dt))
    end
    model
end

function gen_lift_model(X_load)
      model = Model[]

      for k = 1:N-1
        function quadrotor_lift_dynamics!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,params)
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
            Δx = X_load[k][1:3] - x[1:3]
            dir = Δx/norm(Δx)
            ẋ[8:10] = g + (1/m)*(q*F + u[5]*dir) # acceleration in world frame
            ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
            return tau, omega, J, Jinv
        end
        push!(model,midpoint(Model(quadrotor_lift_dynamics!,13,5,quad_params),dt))
    end
    model
end

lift_models = [gen_lift_model_initial(i) for i = 1:num_lift]

function gen_load_model_initial()
      model = Model[]

      for k = 1:N-1
          function double_integrator_3D_dynamics_load!(ẋ,x,u) where T
              Δx1 = (xlift0[1][1:3] - xload0[1:3])
              Δx2 = (xlift0[2][1:3] - xload0[1:3])
              Δx3 = (xlift0[3][1:3] - xload0[1:3])
              u_slack1 = u[1]*Δx1/norm(Δx1)
              u_slack2 = u[2]*Δx2/norm(Δx2)
              u_slack3 = u[3]*Δx3/norm(Δx3)
              Dynamics.double_integrator_3D_dynamics!(ẋ,x,(u_slack1+u_slack2+u_slack3)/di_mass_load)
          end
        push!(model,midpoint(Model(double_integrator_3D_dynamics_load!,6,3),dt))
    end
    model
end

function gen_load_model(X_lift)
      model = Model[]
      for k = 1:N-1
          function double_integrator_3D_dynamics_load!(ẋ,x,u)
              Δx1 = X_lift[1][k][1:3] - x[1:3]
              Δx2 = X_lift[2][k][1:3] - x[1:3]
              Δx3 = X_lift[3][k][1:3] - x[1:3]

              u_slack1 = u[1]*Δx1/norm(Δx1)
              u_slack2 = u[2]*Δx2/norm(Δx2)
              u_slack3 = u[3]*Δx3/norm(Δx3)
              Dynamics.double_integrator_3D_dynamics!(ẋ,x,(u_slack1+u_slack2+u_slack3)/0.35)
          end
        push!(model,midpoint(Model(double_integrator_3D_dynamics_load!,6,3),dt))
    end
    model
end

load_model = gen_load_model_initial()
load_model isa Vector{Model}

include(joinpath(pwd(),"examples/ADMM/models.jl"))

# Robot sizes
r_lift = 0.275
r_load = 0.2

# Control limits for lift robots
u_lim_l = -Inf*ones(m_lift)
u_lim_u = Inf*ones(m_lift)
u_lim_l[1:4] .= 0.
u_lim_l[5] = 0.
u_lim_u[1:4] .= 12.0/4.0
x_lim_l_lift = -Inf*ones(n_lift)
x_lim_l_lift[3] = 0.

x_lim_l_load = -Inf*ones(n_load)
x_lim_l_load[3] = 0.

u_lim_l_load = -Inf*ones(m_load)

bnd1 = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u)
bnd2 = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u,x_min=x_lim_l_lift)
bnd3 = BoundConstraint(n_load,m_load,x_min=x_lim_l_load,u_min=u_lim_l_load)
bnd4 = BoundConstraint(n_load,m_load,x_min=x_lim_l_load)

# Obstacles
r_cylinder = 0.5

_cyl = []
push!(_cyl,(goal_dist/2.,1.,r_cylinder))
push!(_cyl,(goal_dist/2.,-1.,r_cylinder))

# push!(_cyl,(goal_dist/2.,1.25,r_cylinder))
# push!(_cyl,(goal_dist/2.,-1.25,r_cylinder))

function cI_cylinder_lift(c,x,u)
    for i = 1:length(_cyl)
        c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*r_lift)
    end
end
obs_lift = Constraint{Inequality}(cI_cylinder_lift,n_lift,m_lift,length(_cyl),:obs_lift)

function cI_cylinder_load(c,x,u)
    for i = 1:length(_cyl)
        c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*r_load)
    end
end
obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)



# Initial controls
f1 = (x10[1:3] - xload0[1:3])/norm(x10[1:3] - xload0[1:3])
f2 = (x20[1:3] - xload0[1:3])/norm(x20[1:3] - xload0[1:3])
f3 = (x30[1:3] - xload0[1:3])/norm(x30[1:3] - xload0[1:3])
f_mag = hcat(f1, f2, f3)\[0;0;9.81*di_mass_load]
ff = [f_mag[1]*f1, f_mag[2]*f2, f_mag[3]*f3]

thrust = 9.81*(quad_params.m + di_mass_load/num_lift)/4
ulift = [[thrust;thrust;thrust;thrust;f_mag[i]] for i = 1:num_lift]
ulift_r = [[0;0;0;0;f_mag[i]] for i = 1:num_lift]

uload = vcat(f_mag...)

# initial control mid
xloadm = [goal_dist/2.; 0; xload0[3];0.;0.;0.]
f1m = (x1m[1:3] - xloadm[1:3])/norm(x1m[1:3] - xloadm[1:3])
f2m = (x2m[1:3] - xloadm[1:3])/norm(x2m[1:3] - xloadm[1:3])
f3m = (x3m[1:3] - xloadm[1:3])/norm(x3m[1:3] - xloadm[1:3])
f_magm = hcat(f1m, f2m, f3m)\[0;0;9.81*di_mass_load]
ffm = [f_magm[1]*f1m, f_magm[2]*f2m, f_magm[3]*f3m]

thrustm = 9.81*(quad_params.m + di_mass_load/num_lift)/4
uliftm = [[thrustm;thrustm;thrustm;thrustm;f_mag[i]] for i = 1:num_lift]
uliftm_r = [[0;0;0;0;f_mag[i]] for i = 1:num_lift]

uloadm = vcat(f_mag...)

# U0_liftm = [[uliftm[i] for k = 1:N-1] for i = 1:num_lift]
# Discretization
Nmid = Int(floor(N/2))

U0_lift = [[ulift[i] for k = 1:N-1] for i = 1:num_lift]
U0_load = [uload for k = 1:N-1]

Q_lift = 1.0e-1*Diagonal(ones(n_lift))
Q_lift[1,1] = 1.0e-4
r_control = 1.0e-3*ones(4)
r_slack = ones(1)
R_lift = 1.0*Diagonal([r_control;r_slack])
Qf_lift = 100.0*Diagonal(ones(n_lift))

Q_load = 0.0*Diagonal(ones(n_load))
# Q_load[1,1] = 1.0e-4
R_load = Diagonal([r_slack;r_slack;r_slack])
Qf_load = 0.0*Diagonal(ones(n_load))


obj_lift = [LQRObjective(Q_lift,R_lift,Qf_lift,xliftf[i],N,ulift[i]) for i = 1:num_lift]
obj_load = LQRObjective(Q_load,R_load,Qf_load,xloadf,N,uload)

Q_mid_lift = copy(Q_lift)
for i in (1:3)
    Q_mid_lift[i,i] = 100.
end

Q_mid_load = copy(Q_load)
for i in (1:3)
    Q_mid_load[i,i] = 100.
end

cost_mid_lift = [LQRCost(Q_mid_lift,R_lift,xliftmid[i],uliftm[i]) for i = 1:num_lift]
cost_mid_load = LQRCost(Q_mid_load,R_load,xloadm,uloadm)

for i = 1:num_lift
    obj_lift[i].cost[Nmid] = cost_mid_lift[i]
end
obj_load.cost[Nmid] = cost_mid_load

# Constraints
constraints_lift = []
for i = 1:num_lift
    con = Constraints(N)
    con[1] += bnd1
    for k = 2:N-1
        con[k] += bnd2 + obs_lift
    end
    push!(constraints_lift,copy(con))
end

constraints_load = Constraints(N)
for k = 2:N-1
    constraints_load[k] += bnd3 + obs_load
end
constraints_load[N] += goal_constraint(xloadf) + bnd4

# Create problems
prob_lift = [Problem(quadrotor_lift,
                obj_lift[i],
                U0_lift[i],
                integration=:midpoint,
                constraints=constraints_lift[i],
                x0=xlift0[i],
                xf=xliftf[i],
                N=N,
                dt=dt)
                for i = 1:num_lift]

for i = 1:num_lift
    prob_lift[i].model = lift_models[i]
end

prob_load = Problem(doubleintegrator3D_load,
                obj_load,
                U0_load,
                integration=:midpoint,
                constraints=constraints_load,
                x0=xload0,
                xf=xloadf,
                N=N,
                dt=dt)

prob_load.model = load_model

# Solver options
verbose=true

opts_ilqr = iLQRSolverOptions(verbose=true,
      iterations=500)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=0.001,
    cost_tolerance_intermediate=1.0e-4,
    iterations=30,
    penalty_scaling=2.0,
    penalty_initial=10.0)

include(joinpath(pwd(),"examples/ADMM/methods.jl"))

# solve!(prob_load,opts_ilqr)
# prob_load
# Solve
@time plift_al, pload_al, slift_al, sload_al = solve_admm_1slack(prob_lift,prob_load,n_slack,:parallel,opts_al)
# @time plift_al, pload_al, slift_al, sload_al = solve_admm(prob_lift,prob_load,n_slack,:sequential,opts_al)

max_violation(slift_al[1])
max_violation(slift_al[2])
max_violation(slift_al[3])
max_violation(sload_al)

# Visualize
include(joinpath(pwd(),"examples/ADMM/visualization.jl"))

vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, [[pload_al]; plift_al],_cyl)

idx = [(1:3)...,(8:10)...]
plot(plift_al[1].U,label="")
plot(plift_al[1].X,8:10)
plot(plift_al[1].X,1:3)



output_traj(plift_al[1],idx,joinpath(pwd(),"examples/ADMM/traj0.txt"))
output_traj(plift_al[2],idx,joinpath(pwd(),"examples/ADMM/traj1.txt"))
output_traj(plift_al[3],idx,joinpath(pwd(),"examples/ADMM/traj2.txt"))


# for i = 1:num_lift
#     rollout!(prob_lift[i])
# end
# rollout!(prob_load)
#
# vis = Visualizer()
# open(vis)
# visualize_quadrotor_lift_system(vis, [[prob_load]; prob_lift],_cyl)

# plot(prob.X,1:3)
# plot(prob.X,13 .+ (1:3))
#
kk = 3

Δx = pload_al.X[kk][(1:3)] - plift_al[1].X[kk][1:3]
Δx/norm(Δx)

uu = plift_al[1].U[kk][5]
ul = pload_al.U[kk][1]

uu/norm(uu)
ul/norm(ul)

norm(uu)
norm(ul)

plot(plift_al[1].U,1:4)
plot(plift_al[1].U,5:5)
plot(pload_al.U,1:1)

plot(plift_al[2].U,5:5)
plot(pload_al.U,2:2)

plot(plift_al[3].U,5:5)
plot(pload_al.U,3:3)

#
# plot(plift_al[1].U,1:4)
# plot(plift_al[1].U,4 .+ (1:3))
# plot(pload_al.U,(1:3))
#
# plot(prob.U,(7 + 4) .+ (1:3))
# plot(prob.U,(7*3 + 3) .+ (1:3))
