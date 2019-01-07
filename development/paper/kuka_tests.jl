using MeshCatMechanisms
using LinearAlgebra
using RigidBodyDynamics
using Plots
include("../kuka_visualizer.jl")
model, obj = Dynamics.kuka
n,m = model.n, model.m
nn = m   # Number of positions

function hold_trajectory(solver, mech, q)
    state = MechanismState(mech)
    set_configuration!(state, q)
    vd = zero(state.q)
    u0 = dynamics_bias(state)

    n,m,N = get_sizes(solver)
    if length(q) > m
        throw(ArgumentError("system must be fully actuated to hold an arbitrary position ($(length(q)) should be > $m)"))
    end
    U0 = zeros(m,N)
    for k = 1:N
        U0[:,k] = u0
    end
    return U0
end

# Create Mechanism
kuka = parse_urdf(Dynamics.urdf_kuka,remove_fixed_tree_joints=false)
world = root_frame(kuka)

# Create Visualizer
visuals = URDFVisuals(Dynamics.urdf_kuka)
vis = MechanismVisualizer(kuka, visuals)
open(vis)


############################################
#    Unconstrained Joint Space Goal        #
############################################

# Define objective
x0 = zeros(n)
x0[2] = -pi/2
xf = copy(x0)
xf[1] = pi/4
# xf[2] = pi/2

Q = 1e-4*Diagonal(I,n)
Qf = 250.0*Diagonal(I,n)
R = 1e-5*Diagonal(I,m)/2

tf = 5.0
obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

# Define solver
N = 41
solver = Solver(model,obj_uncon,N=N)

# Generate initial control trajectory
U0 = hold_trajectory(solver, kuka, x0[1:7])
U0[:,end] .= 0
X0 = TrajectoryOptimization.rollout(solver,U0)

# Solve
solver.opts.verbose = true
solver.opts.live_plotting = false
solver.opts.iterations_innerloop = 200
solver.state.infeasible
solver.opts.cost_tolerance = 1e-5
res, stats = solve(solver,U0)
norm(res.X[N]-xf)

J = TrajectoryOptimization.cost(solver,res)
TrajectoryOptimization.cost(solver,res.X,res.U) == J

eval_f = gen_usrfun_ipopt(solver::Solver,:hermite_simpson)[1]
res_d, stats_d = solve_dircol(solver,X0,U0)
J - TrajectoryOptimization.cost(solver,res_d.X,res_d.U)


eval_f(res_d.Z)

set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res.X)
set_configuration!(vis, xf[1:7])
plot(to_array(res.U)')


#####################################
#        WITH TORQUE LIMITS         #
#####################################

# Limit Torques
obj_con = ConstrainedObjective(obj_uncon, u_min=-20,u_max=20)
solver = Solver(model,obj_con,N=N)
solver.opts.verbose = false
solver.opts.penalty_scaling = 100
solver.opts.penalty_initial = 0.1
solver.opts.cost_tolerance = 1e-6
solver.opts.cost_tolerance_intermediate = 1e-2
solver.opts.iterations = 200
solver.opts.bp_reg_initial = 0
U_uncon = to_array(res.U)

# iLQR
res_con, stats_con = solve(solver,U_uncon)
cost(solver,res_con)

# DIRCOL
X0 = rollout(solver,to_array(res.U))
res_con_d, stats_con_d = solve_dircol(solver,X0,to_array(res.U))
cost(solver,res_con_d)

# Visualize
set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res_con_d.X)

plot(to_array(res_con.U)')


####################################
#       END EFFECTOR GOAL          #
####################################
# Run ik to get final configuration to achieve goal
goal = [.5,.24,.9]
ik_res = Dynamics.kuka_ee_ik(kuka,goal)
xf[1:nn] = configuration(ik_res)

# Define objective with new final configuration
obj_ik = LQRObjective(Q, R*1e-6, Qf, tf, x0, xf)

# Solve
solver_ik = Solver(model,obj_ik,N=N)
res_ik, stats_ik = solve(solver_ik,U0)
cost(solver,res_ik)
norm(res_ik.X[N] - xf)
ee_ik = Dynamics.calc_ee_position(kuka,res_ik.X)
norm(ee_ik[N] - goal)

X0 = rollout(solver,U0)
options = Dict("max_iter"=>10000,"tol"=>1e-6)
res_ik_d, stats_ik_d = solve_dircol(solver_ik,X0,U0,options=options)
cost(solver,res_ik_d)



# Plot the goal as a sphere
setelement!(vis,Point3D(world,goal),0.02)
set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res_ik_d.X)


##########################################################
#          End-effector obstacle avoidance               #
##########################################################

# Define obstacles
r = 0.2
pos = [N÷10,N÷3]
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
solver.opts.penalty_initial = 0.001
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



# Collision Bubbles
function kuka_points(kuka,plot=false)
    bodies = [3,4,5,6]
    radii = [0.1,0.12,0.09,0.09]
    ee_body,ee_point = Dynamics.get_kuka_ee(kuka)
    ee_radii = 0.05

    points = Point3D[]
    for (idx,radius) in zip(bodies,radii)
        body = findbody(kuka,"iiwa_link_$idx")
        point = Point3D(default_frame(body),0.,0.,0.)
        plot ? setelement!(vis,point,radius,"body_$idx") : nothing
        push!(points,point)
    end
    setelement!(vis,ee_point,ee_radii,"end_effector")
    push!(points,ee_point)
    push!(radii,ee_radii)
    return points, radii
end

function calc_kuka_points(x::Vector{T},points) where T
    state = state_cache[T]
    set_configuration!(state,x[1:nn])
    [transform(state,point,world).v for point in points]
end

function collision_constraint(c,obstacle,kuka_points,radii) where T
    for (i,kuka_point) in enumerate(kuka_points)
        c[i] = sphere_constraint(obstacle,kuka_point,radii[i]+obstacle[4])
    end
end

function generate_collision_constraint(kuka::Mechanism, circles)
    # Specify points along the arm
    points, radii = kuka_points(kuka)
    num_points = length(points)
    num_obstacles = length(circles)

    function cI_obstacles(c,x,u)
        nn = length(u)

        # Get current world location of points along arm
        arm_points = calc_kuka_points(x[1:nn],points)

        C = reshape(view(c,1:num_points*num_obstacles),num_points,num_obstacles)
        for i = 1:num_obstacles
            c_obstacle = view(C,1:num_points,i)
            collision_constraint(c_obstacle,circles[i],arm_points,radii)
        end

    end
end

# Add more obstacles
circles2 = copy(circles)
push!(circles2,[-0.3,0,0.7,0.2])
push!(circles2,[0.3,0.3,1.0,0.1])
push!(circles2,[0.3,-0.5,0.4,0.15])
addcircles!(vis,circles2)

points,radii = kuka_points(kuka,true)

# Generate constraint function
c = zeros(length(points)*length(circles2))
cI_arm_obstacles = generate_collision_constraint(kuka,circles2)
cI_arm_obstacles(c,x0,zeros(m))

# Formulate and solve problem
costfun = LQRCost(Q,R,Qf,xf)
obj_obs_arm = ConstrainedObjective(obj_ik,cI=cI_arm_obstacles,u_min=-80,u_max=80)
obj_obs_arm.cost.Q = Q
obj_obs_arm.cost.R = R*1e-8
solver = Solver(model, obj_obs_arm, N=41)
solver.opts.verbose = true
res_obs, stats_obs = solve(solver,to_array(res_ik.U))
X = res_obs.X
U = res_obs.U
ee_obs_arm = Dynamics.calc_ee_position(kuka,X)

U0_warm = to_array(U)

# Visualize
set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res_ik.X)
animate_trajectory(vis, res_obs.X, 0.2)

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
