using ForwardDiff, LinearAlgebra, Plots, StaticArrays
const TO = TrajectoryOptimization

na = num_lift = 3
dt = 0.2
n_batch = na*13 + 6
m_batch = na*(4 + 1) + na
N = 51


n_lift = 13
m_lift = 5
load_mass = 0.35
n_load = 6
m_load = num_lift
radius_lift = 0.275
radius_load = 0.2


info = Dict{Symbol,Any}(:quat0=>[(4:7) .+ i for i in 0:n_lift:n_lift*num_lift-1])
model = Model(batch_dynamics!,n_batch,m_batch,info)
model_d = midpoint(model,dt)

model_d.f(rand(n_batch),rand(n_batch),rand(m_batch),0.1)

goal_dist = 10.0
TO.has_quat(model_d)



# Initial conditions
x0_load = [0,0,4/6+0.25]
xf_load = [goal_dist,0,4/6+0.25]
d = 1.55
xlift0, xload0 = get_states(x0_load, n_lift, n_load, num_lift, d, deg2rad(80))
xliftf, xloadf = get_states(xf_load, n_lift, n_load, num_lift, d)

# Concatenate into joint states
x0 = vcat(xlift0...,xload0)
xf = vcat(xliftf...,xloadf)



#~~~~~~~~~~~~~~~~~~~~~ OBJECTIVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~#

q_lift, r_lift, qf_lift = quad_costs(n_lift, m_lift)
q_load, r_load, qf_load = load_costs(n_load, m_load)

# Concatenate costs
Q = Diagonal([repeat(q_lift, num_lift); q_load])
R = Diagonal([repeat(r_lift, num_lift); r_load])
Qf = Diagonal([repeat(qf_lift, num_lift); qf_load])


# determine static forces
ulift, uload = calc_static_forces(xlift0, xload0, lift_params.m, load_mass, num_lift)

u0 = vcat(ulift...,uload)
obj = LQRObjective(Q,R,Qf,xf,N,u0)

# get position at midpoint
Nmid = convert(Int,floor(N/2))+1

ℓ1 = norm(xlift0[3][1:3]-xlift0[1][1:3])
ℓ2 = d
ℓ3 = 0.

rm_load = [goal_dist/2, 0, x0_load[3]]
xm_load = get_quad_locations(rm_load, d, deg2rad(45), num_lift, config=:doorway)
xliftmid = [zeros(n_lift) for i = 1:num_lift]
for i = 1:num_lift
    xliftmid[i][1:3] = xm_load[i]
    xliftmid[i][4] = 1.0
end
xliftmid[1][2] = -0.01
xliftmid[1][3] =  0.01

xm = vcat(xliftmid...,xloadm)

# get objective at midpoint
uliftm, uloadm = calc_static_forces(xliftmid, xloadm, lift_params.m, load_mass, num_lift)
um = vcat(uliftm...,uloadm)

q_lift_mid = copy(q_lift)
q_load_mid = copy(q_load)
q_lift_mid[1:3] .= 100
q_load_mid[1:3] .= 100
Q_mid = Diagonal([repeat(q_lift_mid, num_lift); q_load_mid])

cost_mid = LQRCost(Q_mid,R,xm,um)
obj.cost[Nmid] = cost_mid




#~~~~~~~~~~~~~~~~~~~~~~~~~~ CONSTRAINTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

function distance_constraint(c,x,u=zeros(m_batch))
    xload = x[3*13 .+ (1:3)]
    c[1] = norm(x[1:3] - xload)^2 - d^2
    c[2] = norm(x[13 .+ (1:3)] - xload)^2 - d^2
    c[3] = norm(x[2*13 .+ (1:3)] - xload)^2 - d^2

    return nothing
end

function force_constraint(c,x,u)
    c[1] = u[5] - u[3*5 + 1]
    c[2] = u[10] - u[3*5 + 2]
    c[3] = u[15] - u[3*5 + 3]
    return nothing
end

function collision_constraint(c,x,u=zeros(m_batch))
    x1 = x[1:3]
    x2 = x[13 .+ (1:3)]
    x3 = x[2*13 .+ (1:3)]

    c[1] = circle_constraint(x1,x2[1],x2[2],3*radius_lift)
    c[2] = circle_constraint(x2,x3[1],x3[2],3*radius_lift)
    c[3] = circle_constraint(x3,x1[1],x1[2],3*radius_lift)

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
            c[c_shift] = circle_constraint(x[idx_pos],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*radius_lift)
            c_shift += 1
            n_shift += 13
        end
        c[c_shift] = circle_constraint(x[3*13 .+ (1:3)],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*radius_lift)
        c_shift += 1
    end
end
cyl = Constraint{Inequality}(cI_cylinder,n_batch,m_batch,(num_lift+1)*length(_cyl),:cyl)


# Bound constraints
u_min_lift = [0,0,0,0,-Inf]
u_min_load = zeros(num_lift)
u_max_lift = ones(m_lift)*12/4
u_max_lift[end] = Inf
u_max_load = ones(m_load)*Inf

u_l = [repeat(u_min_lift, num_lift); u_min_load]
u_u = [repeat(u_max_lift, num_lift); u_max_load]

bnd = BoundConstraint(n_batch,m_batch,u_min=u_l,u_max=u_u)

dist_con = Constraint{Equality}(distance_constraint,n_batch,m_batch, num_lift, :distance)
for_con = Constraint{Equality}(force_constraint,n_batch,m_batch, num_lift, :force)
col_con = Constraint{Inequality}(collision_constraint,n_batch,m_batch, 3, :collision)
goal = goal_constraint(xf)

con = Constraints(N)

for k = 1:N-1
    con[k] += dist_con + for_con + bnd + col_con + cyl
end
con[N] +=  goal + col_con  + dist_con


# Solver options
verbose=false

opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=250)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-4,
    iterations=20,
    penalty_scaling=10.0,
    penalty_initial=1.0e-3)


# Create Problem
prob = Problem(model_d,obj,dt=dt,N=N,constraints=con,xf=xf,x0=x0)

# Initial controls
U0 = [u0 for k = 1:N-1]
initial_controls!(prob,U0)

prob = gen_prob_batch(quad_params, load_params, batch=true)
rollout!(prob)


# @btime solve($prob,$opts_al)
@time solve!(prob,opts_al)
max_violation(prob)
TO.findmax_violation(prob)
num_constraints(prob.constraints)
prob.constraints[N]
length.(prob.constraints[N],:terminal)

plot(prob.X,1:3)
plot(prob.X,13 .+ (1:3))

kk = 3

Δx = prob.X[kk][13*3 .+ (1:3)] - prob.X[kk][1:3]
Δx/norm(Δx)

uu = prob.U[kk][4 + 1]
ul = prob.U[kk][3*5 + 1]

uu/norm(uu)
ul/norm(ul)

plot(prob.U,1:4)
plot(prob.U,5:5)
plot(prob.U,5*3 + 1)

plot(prob.U,10:10)
plot(prob.U,15 + 2)

plot(prob.U,15:15)

include(joinpath(pwd(),"examples/ADMM/visualization.jl"))

function visualize(vis,prob,num_lift=3)

    # camera angle
    # settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))
    _cyl = door_obstacles()
    addcylinders!(vis, _cyl, 2.1)
    x0 = prob.x0
    d = norm(x0[1:3] - x0[num_lift*13 .+ (1:3)])

    # intialize system
    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.085
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling
    for i = 1:num_lift
        setobject!(vis["lift$i"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))
        cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d),convert(Float32,0.01))
        setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
    end
    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,0.2)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

    anim = MeshCat.Animation(convert(Int,floor(1.0/prob.dt)))
    for k = 1:prob.N
        MeshCat.atframe(anim,vis,k) do frame
            # cables
            x_load = prob.X[k][3*13 .+ (1:3)]
            for i = 1:num_lift
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
