
using TrajectoryOptimization
using Random
using StaticArrays
using LinearAlgebra
using MeshCat
using TrajOptPlots
using Rotations
using BenchmarkTools
import TrajectoryOptimization: Rotation
const TO = TrajectoryOptimization

max_con_viol = 1.0e-3
T = Float64
verbose = true

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    iterations=300)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=40,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-4,
    constraint_tolerance=max_con_viol,
    penalty_scaling=10.,
    penalty_initial=10.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol,
    solve_type=:feasible)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    R_inf=1.0e-8,
    resolve_feasible_problem=false,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-4)


model = Dynamics.Quadrotor2{UnitQuaternion{Float64,IdentityMap}}(use_rot=false)
n,m = 13,4

if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

function gen_quad_maze(Rot; kwargs...)
    prob = Problems.QuadrotorMaze(Rot; kwargs...)
    solver = ALTROSolver(prob, opts_altro, infeasible=true)
    solver.solver_al.opts.verbose = false
    solver
end


# Initialize visualization
solver = gen_quad_maze(RodriguesParam{Float64}, use_rot=false)
solver = gen_quad_maze(UnitQuaternion{Float64,VectorPart}, costfun=:QuatLQR)
solver = gen_quad_maze(UnitQuaternion{Float64,IdentityMap}, use_rot=:slack, normcon=true)
Z0 = deepcopy(get_trajectory(solver))
opts_ilqr.verbose = false
opts_altro.projected_newton = false
solver.opts.verbose = true
add_cylinders!(vis, solver, height=4, robot_radius=1.0)
visualize!(vis, solver)

# Solve
initial_trajectory!(solver, Z0)
@time solve!(solver)
iterations(solver)
visualize!(vis, solver)
benchmark_solve!(solver)
conSet = get_constraints(solver)

# Plot waypoints
waypoints!(vis, get_model(solver), get_trajectory(solver), length=41)


# Step through solve
prob = Problems.QuadrotorMaze(Rot,use_rot=true, costfun=:Quadratic)
prob.constraints.constraints
opts_ilqr.verbose = false
opts_altro.projected_newton = false
opts_altro.R_inf = 1e-8
opts_al.penalty_initial = 1
solver = ALTROSolver(prob, opts_altro, infeasible=true)

al = solver.solver_al
initialize!(al)
TO.set_tolerances!(al, al.solver_uncon, 1)
visualize!(vis, al)

TO.step!(al)
max_violation(al)
visualize!(vis, al)

function compare_quats(gen_prob; kwargs...)
    # Initial data structure
    data = Dict{Symbol,Vector}(:name=>Symbol[], :costfun=>Symbol[], :iterations=>Int[],
        :rotation=>Symbol[], :time=>Float64[], :cost=>Float64[])

    # Treat quaternion as normal vector
    println("Quat")
    log_solve!(gen_prob(UnitQuaternion{Float64, IdentityMap}, use_rot=false),
        data; name=:Quat, kwargs...)

    # Re-normalize after disretization
    println("ReNorm")
    log_solve!(gen_prob(UnitQuaternion{Float64, ReNorm}, use_rot=false),
        data; name=:ReNorm, kwargs...)

    # Use Unit Norm Constraint
    println("NormCon")
    log_solve!(gen_prob(UnitQuaternion{Float64, IdentityMap}, use_rot=false,
        normcon=true), data; name=:NormCon, kwargs...)

    # Use Unit Norm Constrained with slack
    println("QuatSlack")
    log_solve!(gen_prob(UnitQuaternion{Float64, IdentityMap}, use_rot=:slack,
        normcon=true), data; name=:QuatSlack, kwargs...)

    for rmap in [ExponentialMap, CayleyMap, MRPMap, VectorPart]
        println(rmap)
        for costfun in [:QuatLQR, :Quadratic] #[:Quadratic, :QuatLQR, :ErrorQuad]
            log_solve!(gen_prob(UnitQuaternion{Float64, rmap}, costfun=costfun,
                normcon=false), data; kwargs...)
        end
    end
    return data
end

qdata_maze = compare_quats(gen_quad_maze, samples=1, evals=1)


# Run benchmarks
data_maze = run_all(gen_quad_maze, samples=10, evals=1)

df = DataFrame(data)
df.rots = string.(short_names.(df.rotation))
df.time_per_iter = df.time ./ df.iterations
quats = in([:ExponentialMap, :CayleyMap, :MRPMap, :VectorPart])
bases = in([:IdentityMap, :MRP, :RP, :RPY])


############################################################################################
# PLOT 1: Iterations
############################################################################################

base = df[bases.(df.rotation),:]
quat = df[quats.(df.rotation),:]
qlqr  = quat[quat.costfun .== :QuatLQR,:]
equad = quat[quat.costfun .== :ErrorQuadratic,:]
quad  = quat[quat.costfun .== :Quadratic,:]
coord_base = Coordinates(base.rots, base.iterations)
coord1 = Coordinates(qlqr.rots, qlqr.iterations)
coord2 = Coordinates(equad.rots, equad.iterations)
coord3 = Coordinates(quad.rots, quad.iterations)

p = @pgf Axis(
    {
        ybar,
        ylabel="iterations",
        "enlarge x limits" = 0.20,
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=quad.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    Plot(coord3),
    Plot(coord2),
    Plot(coord1),
    Legend(["Quadratic", "Error Quadratic", "Geodesic"])
)
pgfsave("figs/maze_quat_iters.tikz", p, include_preamble=false)


############################################################################################
# PLOT 2: Time
############################################################################################
df[:,[:rotation,:costfun,:time]]
base = df[bases.(df.rotation),:]
quat = df[quats.(df.rotation),:]
qlqr  = quat[quat.costfun .== :QuatLQR,:]
equad = quat[quat.costfun .== :ErrorQuadratic,:]
quad  = quat[quat.costfun .== :Quadratic,:]
quad_all = df[df.costfun .== :Quadratic,:]
best = [base; qlqr]
p = @pgf Axis(
    {
        ybar,
        ylabel="solve time (ms)",
        # "enlarge limits" = 0.20,
        legend_style =
        {
            at = Coordinate(0.5, -0.07),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=best.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
        "every node near coord/.append style={/pgf/number format/.cd, fixed,precision=0}"
    },
    Plot(Coordinates(best.rots, best.time)),
    # Legend(["Quadratic", "Error Quadratic", "Geodesic"])
)
pgfsave("figs/maze_time_best.tikz", p, include_preamble=false)

coord_base = Coordinates(base.rots, base.time)
coord_all = Coordinates(quad_all.rots, quad_all.time)
coord1 = Coordinates(qlqr.rots, qlqr.time)
coord2 = Coordinates(equad.rots, equad.time)
coord3 = Coordinates(quad.rots, quad.time)


p = @pgf Axis(
    {
        ybar,
        width="20cm",
        ylabel="solve time (ms)",
        # "enlarge limits" = 0.20,
        legend_style =
        {
            at = Coordinate(0.5, -0.07),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=quad_all.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
        "every node near coord/.append style={/pgf/number format/.cd, fixed,precision=0}"
    },
    Plot(coord_all),
    Plot(coord2),
    Plot(coord1),
    Legend(["Quadratic", "Error Quadratic", "Geodesic"])
)
pgfsave("figs/maze_time_all.tikz", p, include_preamble=false)


############################################################################################
# PLOT 4: Quat method comparison - time
############################################################################################

df = DataFrame(data_quats)
df.rots = string.(short_names.(df.name))
quats = by(df, :rots, :time=>minimum)


p = @pgf Axis(
    {
        ybar,
        ylabel="solve time (ms)",
        # "enlarge limits" = 0.20,
        legend_style =
        {
            at = Coordinate(0.5, -0.07),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=quats.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
        "every node near coord/.append style={/pgf/number format/.cd, fixed,precision=0}"
    },
    # HLine({"ultra thick", "white"}, 35),
    Plot(Coordinates(quats.rots, quats.time_minimum)),
    # Legend(["Quadratic", "Error Quadratic", "Geodesic"])
)
pgfsave("figs/maze_qcomp.tikz", p, include_preamble=false)
