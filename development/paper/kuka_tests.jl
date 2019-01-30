using MeshCatMechanisms
using LinearAlgebra
using RigidBodyDynamics
using Plots
import TrajectoryOptimization: hold_trajectory, Trajectory, total_time
import RigidBodyDynamics: transform
include("N_plots.jl")?
include("../kuka_visualizer.jl")
model, obj = Dynamics.kuka
n,m = model.n, model.m
nn = m   # Number of positions

function plot_sphere(frame::CartesianFrame3D,center,radius,mat,name="")
    geom = HyperSphere(Point3f0(center), convert(Float32,radius))
    setelement!(vis,frame,geom,mat,name)
end

function plot_cylinder(frame::CartesianFrame3D,c1,c2,radius,mat,name="")
    geom = Cylinder(Point3f0(c1),Point3f0(c2),convert(Float32,radius))
    setelement!(vis,frame,geom,mat,name)
end


# Create Mechanism
kuka = parse_urdf(Dynamics.urdf_kuka,remove_fixed_tree_joints=false)
world = root_frame(kuka)

# Create Visualizer
visuals = URDFVisuals(Dynamics.urdf_kuka)
vis = MechanismVisualizer(kuka, visuals)
open(vis)


# Default solver options
opts = SolverOptions()
opts.cost_tolerance = 1e-6
opts.constraint_tolerance = 1e-5
dircol_options = Dict("tol"=>opts.cost_tolerance,"constr_viol_tol"=>opts.constraint_tolerance)


############################################
#    Unconstrained Joint Space Goal        #
############################################

# Define objective
x0 = zeros(n)
x0[2] = -pi/2
xf = copy(x0)
xf[1] = pi/4
Q = 1e-4*Diagonal(I,n)*10
Qf = 250.0*Diagonal(I,n)
R = 1e-5*Diagonal(I,m)/100
Rd = 1e-6
R = Diagonal([1e-8,1e-8,Rd,Rd,Rd,Rd,Rd])
tf = 5.0
obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

# Define solver
N = 41
solver = Solver(model,obj_uncon,N=N, opts=opts)

# Generate initial control trajectory
U0_hold = hold_trajectory(solver, kuka, x0[1:7])
U0_hold[:,end] .= 0
X0_hold = TrajectoryOptimization.rollout(solver,U0_hold)

# Solve
solver.opts.verbose = true
solver.opts.live_plotting = false
solver.opts.iterations_innerloop = 200
solver.state.infeasible
solver.opts.bp_reg_initial = 0
res, stats = solve(solver,U0_hold)
stats["iterations"]
J = stats["cost"][end]
norm(res.X[N]-xf)
plot(res.U)

res_d, stats_d = solve_dircol(solver,X0_hold,U0_hold,options=dircol_options)
eval_f = gen_usrfun_ipopt(solver::Solver,:hermite_simpson)[1]
J - cost(solver,res_d.X,res_d.U)

set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res.X)
set_configuration!(vis, xf[1:7])
animate_trajectory(vis, res_d.X)

plot(stats["cost"],yscale=:log10)
plot!(stats_d["cost"])

u_bnd = [50,100,30,50,30,30,30]
obj_con = ConstrainedObjective(obj_uncon, u_min=-u_bnd, u_max=u_bnd)
solver_con = Solver(model, obj_con, N=N, opts=opts)
res_con, stats_con = solve(solver_con, U0_hold)


opts = SolverOptions()
opts.square_root = true
opts.R_minimum_time = 40
opts.minimum_time_tf_estimate = 0.6
opts.penalty_initial = 1
opts.penalty_scaling = 20
opts.penalty_initial_minimum_time_equality = 10
opts.penalty_initial_minimum_time_inequality = 10
opts.outer_loop_update_type = :default
opts.cost_tolerance = 1e-4
opts.constraint_tolerance = 1e-3
opts.cost_tolerance_intermediate = 1e-2
opts.use_nesterov = false

obj_min = update_objective(obj_con, tf=:min)
solver_min = Solver(model, obj_min, N=N, opts=opts)
solver_min.opts.verbose = true
res_min, stats_min = solve(solver_min,to_array(res_con.U))
stats_min["iterations"]
TrajectoryOptimization.total_time(solver,res)
TrajectoryOptimization.total_time(solver_min,res_min)
plot(res_min.X,legend=:none)
plot(res_min.U)
plot(res_min.X, 1:6)
plot(res.U)
plot!(res_min.U)


dt = [u[end] for u in res_min.U]
set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res.X, solver.dt)
animate_trajectory(vis, res_min.X, mean(dt)^2)

plot(dt)

#####################################
#        WITH TORQUE LIMITS         #
#####################################

# Limit Torques
obj_con = ConstrainedObjective(obj_uncon, u_min=-75,u_max=75)
U_uncon = to_array(res.U)

solver = Solver(model,obj_con,N=N, opts=opts)
solver.opts.verbose = false
solver.opts.penalty_scaling = 50
solver.opts.penalty_initial = 0.0001
solver.opts.cost_tolerance_intermediate = 1e-3
solver.opts.outer_loop_update_type = :feedback
solver.opts.use_nesterov = false
solver.opts.iterations = 500
solver.opts.bp_reg_initial = 0

# iLQR
res_con, stats_con = solve(solver,U0_hold)
stats_con["iterations"]
cost(solver,res_con)

# DIRCOL
res_con_d, stats_con_d = solve_dircol(solver,X0_hold,U0_hold,options=dircol_options)
cost(solver,res_con_d)

p = convergence_plot(stats_con,stats_con_d,xscale=:log10)
TrajectoryOptimization.plot_vertical_lines!(p,stats_con["outer_updates"],title="Kuka Arm with Torque Limits")

# Visualize
set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res_con.X)

# Dircol Truth
group = "kuka/armswing/constrained"
solver_truth = Solver(model,obj_con,N=101)
run_dircol_truth(solver_truth, Array(res_con_d.X), Array(res_con_d.U), group::String)
X_truth, U_truth = get_dircol_truth(solver_truth,X0_hold,U0_hold,group)[2:3]
interp(t) = TrajectoryOptimization.interpolate_trajectory(solver_truth, X_truth, U_truth, t)

Ns = [41,51,75,101]
disable_logging(Logging.Debug)
run_step_size_comparison(model, obj_con, U0_hold, group, Ns, integrations=[:midpoint,:rk3],dt_truth=solver_truth.dt,opts=solver.opts,X0=X0_hold)
plot_stat("runtime",group,legend=:bottomright,title="Kuka Arm with Torque Limits")
plot_stat("iterations",group,legend=:bottom,title="Kuka Arm with Torque Limits")
plot_stat("error",group,yscale=:log10,legend=:left,title="Kuka Arm with Torque Limits")
plot_stat("c_max",group,yscale=:log10,legend=:bottom,title="Kuka Arm with Torque Limits")


####################################
#       END EFFECTOR GOAL          #
####################################

# Run ik to get final configuration to achieve goal
goal = [.5,.24,.9]
ik_res = Dynamics.kuka_ee_ik(kuka,goal)
xf[1:nn] = configuration(ik_res)

# Define objective with new final configuration
obj_ik = LQRObjective(Q, R, Qf, tf, x0, xf)

# Solve
solver_ik = Solver(model,obj_ik,N=N)
res_ik, stats_ik = solve(solver_ik,U0_hold)
stats_ik["iterations"]
cost(solver,res_ik)
norm(res_ik.X[N] - xf)
ee_ik = Dynamics.calc_ee_position(kuka,res_ik.X)
norm(ee_ik[N] - goal)

X0 = rollout(solver,U0_hold)
options = Dict("max_iter"=>10000,"tol"=>1e-6)
res_ik_d, stats_ik_d = solve_dircol(solver_ik,X0_hold,U0_hold,options=options)
cost(solver,res_ik_d)

# Plot the goal as a sphere
plot_sphere(world,goal,0.02,green_,"goal")
set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res_ik.X)


##########################################################
#          End-effector obstacle avoidance               #
##########################################################

# Define obstacles
r = 0.2
pos = Int.(floor.([0.12N, 0.28N]))
circles = [ee_ik[p] for p in pos]
for circle in circles; push!(circle,r) end
n_obstacles = length(circles)

import TrajectoryOptimization.sphere_constraint
state_cache = StateCache(kuka)
ee_body, ee_point = Dynamics.get_kuka_ee(kuka)
world = root_frame(kuka)

function cI(c,x::AbstractVector{T},u) where T
    state = state_cache[T]
    set_configuration!(state,x[1:nn])
    ee = transform(state,ee_point,world).v
    for i = 1:n_obstacles
        c[i] = sphere_constraint(ee,circles[i][1],circles[i][2],circles[i][3],r)
    end
end

function addcircles!(vis,circles)
    world = root_frame(kuka)
    for (i,circle) in enumerate(circles)
        p = Point3D(world,collect(circle[1:3]))
        setelement!(vis,p,circle[4],"obs$i")
    end
end

# Plot Obstacles in Visualizer
addcircles!(vis,circles)

# Formulate and solve problem
obj_obs = ConstrainedObjective(obj_ik,cI=cI)
solver = Solver(model, obj_obs, N=N)
solver.opts.verbose = false
solver.opts.penalty_scaling = 100
solver.opts.penalty_initial = 0.0001
# solver.opts.cost_tolerance = 1e-6
solver.opts.cost_tolerance_intermediate = 1e-3
# solver.opts.iterations = 200
solver.opts.bp_reg_initial = 10
U0 = to_array(res_ik.U)
U0_hold = hold_trajectory(solver,kuka,x0[1:nn])
res_obs, stats_obs = solve(solver,U0_hold)
cost(solver,res_obs)
X = res_obs.X
U = res_obs.U
ee_obs = Dynamics.calc_ee_position(kuka,X)

X0 = rollout(solver,U0_hold)
res_obs_d, stat_obs_d = solve_dircol(solver,X0,U0,options=options)
cost(solver,res_obs_d)

# Visualize
set_configuration!(vis, x0[1:7])
# animate_trajectory(vis, res_ik.X)
animate_trajectory(vis, res_obs.X, 0.2)

# Plot Constraints
C0 = zero.(res_obs.C)
for k = 1:N
    cI(C0[k],res_ik.X[k],res_ik.U[k])
end
cI(res_obs.C[N],X[N],U[N])
plot(to_array(C0)',legend=:bottom, width=2, color=[:blue :red :green :yellow])
plot!(to_array(res_obs.C)',legend=:bottom, color=[:blue :red :green :yellow])

# Compare EE trajectories
plot(to_array(ee_ik)',width=2)
plot!(to_array(ee_obs)')


##########################################################
#            Full body Obstacle Avoidance                #
##########################################################

# Collision Bubbles
function kuka_points(kuka,plot=false)
    bodies = [3,4,5,6]
    radii = [0.1,0.12,0.09,0.09]
    ee_body,ee_point = Dynamics.get_kuka_ee(kuka)
    ee_radii = 0.05

    points = Point3D[]
    frames = CartesianFrame3D[]
    for (idx,radius) in zip(bodies,radii)
        body = findbody(kuka,"iiwa_link_$idx")
        frame = default_frame(body)
        point = Point3D(frame,0.,0.,0.)
        plot ? plot_sphere(frame,0,radius,body_collision,"body_$idx") : nothing
        # plot ? setelement!(vis,point,radius,"body_$idx") : nothing
        push!(points,point)
        push!(frames,frame)
    end
    frame = default_frame(ee_body)
    plot ? plot_sphere(frame,0,ee_radii,body_collision,"end_effector") : nothing
    # plot ? setelement!(vis,ee_point,ee_radii,"end_effector") : nothing
    push!(points,ee_point)
    push!(radii,ee_radii)
    return points, radii, frames
end

function calc_kuka_points(x::Vector{T},points) where T
    state = state_cache[T]
    set_configuration!(state,x[1:nn])
    [transform(state,point,world).v for point in points]
end

function collision_constraint(c,obstacle,kuka_points,radii) where T

end

function generate_collision_constraint(kuka::Mechanism, circles, cylinders=[])
    # Specify points along the arm
    points, radii = kuka_points(kuka)
    num_points = length(points)
    nCircle = length(circles)
    nCylinder = length(cylinders)
    num_obstacles = nCircle + nCylinder

    function cI_obstacles(c,x,u)
        nn = length(u)

        # Get current world location of points along arm
        arm_points = calc_kuka_points(x[1:nn],points)

        C = reshape(view(c,1:num_points*num_obstacles),num_points,num_obstacles)
        for i = 1:nCircle
            c_obstacle = view(C,1:num_points,i)
            for (p,kuka_point) in enumerate(arm_points)
                c_obstacle[p] = sphere_constraint(circles[i],kuka_point,radii[p]+circles[i][4])
            end
        end
        for j = 1:nCylinder
            i = j + nCircle
            c_obstacle = view(C,1:num_points,i)
            for (p,kuka_point) in enumerate(arm_points)
                c_obstacle[p] = circle_constraint(cylinders[j],kuka_point,radii[p]+cylinders[j][3])
            end
        end
    end
end

function addcylinders!(vis,cylinders,height=1.5)
    for (i,cyl) in enumerate(cylinders)
        plot_cylinder(world,[cyl[1],cyl[2],0],[0,0,height],cyl[3],blue_,"cyl_$i")
    end
end

# Define objective
x0 = zeros(n)
x0[2] = pi/2
x0[3] = pi/2
x0[4] = pi/2
xf = zeros(n)
xf[1] = pi/2
xf[4] = pi/2
set_configuration!(vis, xf[1:7])
Q = 1e-4*Diagonal(I,n)*10
Qf = 250.0*Diagonal(I,n)
R = 1e-5*Diagonal(I,m)/100
Rd = 1e-6
R = Diagonal([1e-8,1e-8,Rd,Rd,Rd,Rd,Rd])
tf = 5.0
obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)


set_configuration!(vis, x0[1:7])

# Add more obstacles
d = 0.25
circles2 = copy(circles)
circles2 = Vector{Float64}[]
push!(circles2,[d,0.0,1.2,0.2])
push!(circles2,[0,-d,0.4,0.2])
push!(circles2,[0,-d,1.2,0.2])
addcircles!(vis,circles2)

cylinders = [[d,-d,0.1],[d,d,0.1],[-d,-d,0.1]]
addcylinders!(vis,cylinders)

points,radii,frames = kuka_points(kuka,true)


# Generate constraint function
num_obstacles = length(circles2)+length(cylinders)
c = zeros(length(points)*num_obstacles)
cI_arm_obstacles = generate_collision_constraint(kuka,circles2,cylinders)
cI_arm_obstacles(c,x0,zeros(m))
c

# Formulate and solve problem
costfun = LQRCost(Q,R,Qf,xf)
obj_obs_arm = ConstrainedObjective(costfun,tf,x0,xf,cI=cI_arm_obstacles,u_min=-80,u_max=80)
obj_obs_arm.cost.Q = Q
obj_obs_arm.cost.R = R*1e-8

solver = Solver(model, obj_obs_arm, N=41)
solver.opts.verbose = true
solver.opts.penalty_scaling = 10
solver.opts.penalty_initial = 0.05
# solver.opts.cost_tolerance = 1e-6
solver.opts.cost_tolerance_intermediate = 1e-2
solver.opts.cost_tolerance = 1e-7
# solver.opts.iterations = 200
solver.opts.bp_reg_initial = 0
solver.opts.outer_loop_update_type = :default
U0_hold = hold_trajectory(solver, kuka, x0[1:7])
res_obs, stats_obs = solve(solver,U0_hold)
X = res_obs.X
U = res_obs.U
ee_obs_arm = Dynamics.calc_ee_position(kuka,X)

U0_warm = to_array(U)

# Visualize
set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res_ik.X)
animate_trajectory(vis, res_obs.X, 0.2)

X_interp, U_interp = TrajectoryOptimization.interp_traj(201,5.,res_obs.X,res_obs.U)
animate_trajectory(vis, X_interp, 0.05)

interpolate_trajectory

plot(to_array(res_obs.U)',legend=:none)

plot(to_array(res_obs.C)')

point_positions = [[zeros(3) for i = 1:length(points)] for k = 1:N]
for k = 1:N
    point_pos = calc_kuka_points(res_obs.X[k][1:nn],points)
    for i = 1:length(points)
        point_positions[k][i] = point_pos[i]
    end
end

dist = zeros(N)
for k = 1:N
    dist[k[]] = sphere_constraint(point_positions[k][2],circles2[3],circles2[3][4]+radii[2])
end
point_positions[2][1]

plot(dist)

c = zeros(5)
collision_constraint(c,circles2[3],point_positions[5],radii)
c

c = zeros(5*5)
cI_arm_obstacles(c,res_obs.X[5],res_obs.U[5])



obj_min = update_objective(obj_obs_arm,tf=0.6)
solver_min = Solver(model,obj_min,N=N,opts=opts)
solver_min.opts.penalty_scaling = 150
solver_min.opts.penalty_initial = 0.0005
# solver.opts.cost_tolerance = 1e-6
solver_min.opts.cost_tolerance_intermediate = 1e-2
solver_min.opts.cost_tolerance = 1e-4
solver_min.opts.constraint_tolerance = 1e-3
# solver.opts.iterations = 200
solver.opts.bp_reg_initial = 0

solver_min.opts.verbose = true
res_min0,stats_min0 = solve(solver_min,U0_hold)

X_interp, U_interp = TrajectoryOptimization.interp_traj(201,5.,res_min.X,res_min.U)
animate_trajectory(vis, X_interp, solver_min.dt)


obj_min = update_objective(obj_obs_arm,tf=:min)
obj_min.cost.Q .= Diagonal([ones(nn)*1e-4; ones(nn)*1e-1])
obj_min.cost.R .= Diagonal(I,m)*1e-6

solver_min = Solver(model,obj_min,N=N,opts=opts)
solver_min.opts.penalty_scaling = 10
solver_min.opts.penalty_initial = 0.0005
# solver.opts.cost_tolerance = 1e-6
solver_min.opts.cost_tolerance_intermediate = 1e-3
solver_min.opts.cost_tolerance = 1e-4
solver_min.opts.constraint_tolerance = 1e-3
solver_min.opts.minimum_time_tf_estimate = 0.6
solver_min.opts.penalty_initial_minimum_time_equality = 10
solver_min.opts.penalty_initial_minimum_time_inequality = 10
solver_min.opts.penalty_scaling_minimum_time_equality = 100
solver_min.opts.R_minimum_time = 1
solver_min.opts.square_root = true
solver_min.opts.use_nesterov = false
solver_min.opts.iterations = 1000

solver_min.opts.verbose = true
U0 = hold_trajectory(solver_min, kuka, x0[1:7])
U_warm = to_array(res_min0.U)
res_min,stats_min = solve(solver_min, U_warm)
total_time(solver_min,res_min)

U_guess = to_array(res_min.U)
res_min,stats_min = solve(solver_min, U_guess)

dt = [u[end] for u in res_min.U]
plot(dt)
plot(res_min.U)
plot(res_min.X,1:6,legend=:bottom)


X_interp, U_interp = TrajectoryOptimization.interp_traj(201,5.,res_min.X,res_min.U)
animate_trajectory(vis, X_interp, 0.01)
solver_min.dt


##########################################################
#          End-effector frame Cost Funtion               #
##########################################################

function get_ee_costfun(mech,Qee,Q,R,Qf,ee_goal)
    statecache = StateCache(mech)
    state = MechanismState(mech)
    world = root_frame(mech)
    ee_pos = Dynamics.get_kuka_ee_postition_fun(mech,statecache)
    nn = num_positions(mech)
    n,m = 2nn,nn

    ee_body, ee_point = Dynamics.get_kuka_ee(kuka)
    p = path(mech, root_body(mech), ee_body)
    Jp = point_jacobian(state, p, transform(state, ee_point, world))

    # Hessian of the ee term wrt full state
    hess_ee = zeros(n,n)
    grad_ee = zeros(n)
    linds = LinearIndices(hess_ee)
    qq = linds[1:nn,1:nn]
    q_inds = 1:nn

    H = zeros(m,n)

    function costfun(x,u)
        ee = ee_pos(x) - ee_goal
        return (ee'Qee*ee + x'Q*x + u'R*u)/2
    end
    function costfun(xN::Vector{Float64})
        ee = ee_pos(xN) - ee_goal
        return 0.5*ee'Qf*ee + xN'Q*xN
    end

    function expansion(x::Vector{T},u) where T
        state = statecache[T]
        set_configuration!(state,x[1:nn])
        ee = ee_pos(x)

        point_in_world = transform(state, ee_point, world)
        point_jacobian!(Jp, state, p, point_in_world)
        Jv = Array(Jp)

        hess_ee[qq] = Jv'Qee*Jv
        Pxx = hess_ee + Q
        Puu = R
        Pux = H

        grad_ee[q_inds] = Jv'Qee*(ee-ee_goal)
        Px = grad_ee + Q*x
        Pu = R*u
        return Pxx,Puu,Pux,Px,Pu
    end
    function expansion(xN::AbstractVector{T}) where T
        state = statecache[T]
        set_configuration!(state,xN[1:nn])
        ee = ee_pos(xN)

        point_in_world = transform(state, ee_point, world)
        point_jacobian!(Jp, state, p, point_in_world)
        Jv = Array(Jp)

        hess_ee[qq] = Jv'Qf*Jv
        Pxx = hess_ee + Q

        grad_ee[q_inds] = Jv'Qf*(ee-ee_goal)
        Px = grad_ee + Q*xN
        return Pxx,Px
    end
    return costfun, expansion
end

Q2 = Diagonal([zeros(nn); ones(nn)*1e-5])
Qee = Diagonal(I,3)*1e-3
Qfee = Diagonal(I,3)*200
R2 = Diagonal(I,m)*1e-12

u0 = zeros(m)
eecostfun, expansion = get_ee_costfun(kuka,Qee,Q2,R2,Qfee,goal)
ee_costfun = GenericCost(eecostfun,eecostfun,expansion,n,m)
stage_cost(ee_costfun,x0,u0)
expansion(x0,u0)
expansion(x0)
TrajectoryOptimization.taylor_expansion(ee_costfun,x0)

ee_obj = UnconstrainedObjective(ee_costfun,tf,x0)
u_bnd = ones(m)*20
u_bnd[2] = 100
u_bnd[4] = 30
x_bnd = ones(n)*Inf
x_bnd[nn+1:end] .= 15
ee_obj_con = ConstrainedObjective(ee_obj,u_min=-u_bnd,u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd,use_xf_equality_constraint=false)

U0_ik = to_array(res_ik.U)
solver_ee = Solver(model,ee_obj_con,N=61)
solver_ee.opts.cost_tolerance = 1e-6
solver_ee.opts.verbose = true
res,stats = solve(solver_ee,U0_hold)

set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res_ik.X)
animate_trajectory(vis, res.X)

plot(res.U,legend=:bottom)
qdot = to_array(res.X)[8:end,:]
plot(qdot')

res.C[N]
to_array(res.X)
res.X[end]
