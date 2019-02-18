using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using Random

Random.seed!(123)

# Solver options
N = 101 # 201
integration = :rk4
opts = SolverOptions()
opts.verbose = false
opts.square_root = true
opts.cost_tolerance = 1e-5
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-4
opts.outer_loop_update_type = :feedback

dircol_options = Dict("tol"=>opts.cost_tolerance,"constr_viol_tol"=>opts.constraint_tolerance)

# Obstacle Avoidance
model,obj_uncon = TrajectoryOptimization.Dynamics.quadrotor
r_quad = 3.0
n = model.n
m = model.m
obj_con = TrajectoryOptimization.Dynamics.quadrotor_3obs[2]
spheres = TrajectoryOptimization.Dynamics.quadrotor_3obs[3]
n_spheres = length(spheres)

solver_uncon = Solver(model,obj_uncon,integration=integration,N=N,opts=opts)
solver_con = Solver(model,obj_con,integration=integration,N=N,opts=opts)

U_hover = 0.5*9.81/4.0*ones(solver_uncon.model.m, solver_uncon.N-1)
X_hover = rollout(solver_uncon,U_hover)
@time results_uncon, stats_uncon = solve(solver_uncon,U_hover)
# @time results_uncon_dircol, stats_uncon_dircol = TrajectoryOptimization.solve_dircol(solver_uncon, X_hover, U_hover, options=dircol_options)

@time results_con, stats_con = solve(solver_con,U_hover)
# @time results_dircol, stats_dircol = TrajectoryOptimization.solve_dircol(solver_con, X_hover, U_hover, options=dircol_options)

# Trajectory plots
t_array = range(0,stop=solver_con.obj.tf,length=solver_con.N)
plot(t_array[1:end-1],to_array(results_uncon.U)',title="Quadrotor Obstacle Avoidance",xlabel="time",ylabel="control",labels="")
plot(t_array,to_array(results_con.X)[1:3,:]',title="Quadrotor Obstacle Avoidance",xlabel="time",ylabel="position",labels=["x";"y";"z"],legend=:topleft)

# plot(t_array[1:end-1],Array(results_uncon_dircol.U)[:,1:end-1]',title="Quadrotor Obstacle Avoidance",xlabel="time",ylabel="control",labels="dircol")
# plot(t_array,Array(results_uncon_dircol.X)[1:3,:]',title="Quadrotor Obstacle Avoidance",xlabel="time",ylabel="control",labels="dircol")

t_array = range(0,stop=solver_con.obj.tf,length=solver_con.N)
plot(t_array[1:end-1],Array(results_dircol.U)[:,1:end-1]',title="Quadrotor Obstacle Avoidance",xlabel="time",ylabel="control",labels="dircol")
plot(t_array,Array(results_dircol.X)[1:3,:]',title="Quadrotor Obstacle Avoidance",xlabel="time",ylabel="position",labels=["x";"y";"z"],legend=:topleft)
@assert max_violation(results_con) <= opts.constraint_tolerance

# Constraint convergence plot
plot(log.(stats_con["c_max"]),title="Quadrotor Obstacle Avoidance",xlabel="iteration",ylabel="log(max constraint violation)",label="")

@show stats_con["iterations"]
@show stats_con["outer loop iterations"]
@show stats_con["c_max"][end]
@show stats_con["cost"][end]
