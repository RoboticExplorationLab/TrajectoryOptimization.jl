using TrajectoryOptimization
using Random
using BenchmarkTools
Random.seed!(7)
### Solver options ###
dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = false
opts.cache = false
opts.c1 = 1e-8
opts.c2 = 10.0
opts.cost_tolerance_intermediate = 1e-4
opts.constraint_tolerance = 1e-4
opts.cost_tolerance = 1e-4
opts.iterations_outerloop = 50
opts.iterations = 250
opts.iterations_linesearch = 25
opts.constraint_decrease_ratio = 0.25
opts.penalty_scaling = 10.0
opts.bp_reg_initial = 1.0
opts.outer_loop_update_type = :individual
opts.use_static = false
opts.resolve_feasible = false
opts.λ_second_order_update = false
opts.bp_reg_type = :state
######################

### Set up model, objective, solver ###
# Model, objective (unconstrained)
model_pendulum, obj_uncon_pendulum = TrajectoryOptimization.Dynamics.pendulum!
model_dubins, obj_uncon_dubins = TrajectoryOptimization.Dynamics.dubinscar!
model_cartpole, obj_uncon_cartpole = TrajectoryOptimization.Dynamics.cartpole_udp

## Constraints

# pendulum
u_min_pendulum = -2
u_max_pendulum = 2
x_min_pendulum = [-20;-20]
x_max_pendulum = [20; 20]

# dubins car
u_min_dubins = [-1; -1]
u_max_dubins = [1; 1]
x_min_dubins = [0; -100; -100]
x_max_dubins = [1.0; 100; 100]

# cartpole
u_min_cartpole = -20
u_max_cartpole = 40
x_min_cartpole = [-10; -1000; -1000; -1000]
x_max_cartpole = [10; 1000; 1000; 1000]

# -Constrained objective
obj_con_pendulum = ConstrainedObjective(obj_uncon_pendulum, u_min=u_min_pendulum, u_max=u_max_pendulum, x_min=x_min_pendulum, x_max=x_max_pendulum)
obj_con_dubins = ConstrainedObjective(obj_uncon_dubins, u_min=u_min_dubins, u_max=u_max_dubins, x_min=x_min_dubins, x_max=x_max_dubins)
obj_con_cartpole = ConstrainedObjective(obj_uncon_cartpole, u_min=u_min_cartpole, u_max=u_max_cartpole, x_min=x_min_cartpole, x_max=x_max_cartpole)

# Solver
intergrator = :rk3
solver_pendulum = Solver(model_pendulum,obj_con_pendulum,integration=intergrator,dt=dt,opts=opts)
solver_dubins = Solver(model_dubins,obj_con_dubins,integration=intergrator,dt=dt,opts=opts)
solver_cartpole = Solver(model_cartpole,obj_con_cartpole,integration=intergrator,dt=dt,opts=opts)

# -Initial state and control trajectories
X_interp_pendulum = line_trajectory(solver_pendulum.obj.x0,solver_pendulum.obj.xf,solver_pendulum.N)
X_interp_dubins = line_trajectory(solver_dubins.obj.x0,solver_dubins.obj.xf,solver_dubins.N)
X_interp_cartpole = line_trajectory(solver_cartpole.obj.x0,solver_cartpole.obj.xf,solver_cartpole.N)

U_pendulum = rand(solver_pendulum.model.m,solver_pendulum.N)
U_dubins = rand(solver_dubins.model.m,solver_dubins.N)
U_cartpole = rand(solver_cartpole.model.m,solver_cartpole.N)

#######################################

### Solve ###
@time results_pendulum, stats_pendulum = solve(solver_pendulum,U_pendulum)
@time results_dubins, stats_dubins = solve(solver_dubins,U_dubins)
@time results_cartpole, stats_cartpole = solve(solver_cartpole,U_cartpole)

function cost_constraints_thread(solver::Solver, res::ConstrainedIterResults)
    N = solver.N
    J = zeros(N)
    Threads.@threads for k = 1:N-1
        J[k] = 0.5*res.C[k]'*res.Iμ[k]*res.C[k] + res.LAMBDA[k]'*res.C[k]
    end

    if solver.control_integration == :foh
        J[N] = 0.5*res.C[N]'*res.Iμ[N]*res.C[N] + res.LAMBDA[N]'*res.C[N]
    end

    J[N] += 0.5*res.CN'*res.IμN*res.CN + res.λN'*res.CN

    return sum(J)
end

function _cost_thread(solver::Solver,res::SolverVectorResults,X=res.X,U=res.U)
    # pull out solver/objective values
    N = solver.N; Q = solver.obj.Q; xf::Vector{Float64} = solver.obj.xf; Qf::Matrix{Float64} = solver.obj.Qf; m = solver.model.m; n = solver.model.n
    obj = solver.obj
    dt = solver.dt

    R = getR(solver)

    J = zeros(N)
    Threads.@threads for k = 1:N-1
        if solver.control_integration == :foh
            Xm = res.xmid[k]
            Um = (U[k] + U[k+1])/2

            J[k] = solver.dt/6*(stage_cost(X[k],U[k],Q,R,xf) + 4*stage_cost(Xm,Um,Q,R,xf) + stage_cost(X[k+1],U[k+1],Q,R,xf)) # Simpson quadrature (integral approximation) for foh stage cost
        else
            J[k] = solver.dt*stage_cost(X[k],U[k],Q,R,xf)
        end
    end

    J[N] = 0.5*(X[N] - xf)'*Qf*(X[N] - xf)

    return sum(J)
end

# threaded functions
function calculate_midpoints_thread!(results::SolverVectorResults, solver::Solver, X::Vector, U::Vector)
    n,m,N = get_sizes(solver)
    Threads.@threads for k = 1:N-1
        results.xmid[k] = cubic_midpoint(results.X[k],results.xdot[k],results.X[k+1],results.xdot[k+1],solver.dt)
    end
end

function calculate_derivatives_thread!(results::SolverVectorResults, solver::Solver, X::Vector, U::Vector)
    n,m,N = get_sizes(solver)
    Threads.@threads for k = 1:N
        solver.fc(results.xdot[k],X[k],U[k][1:m])
    end
end

function calculate_jacobians_thread!(res::ConstrainedIterResults, solver::Solver)::Nothing #TODO change to inplace '!' notation throughout the code
    N = solver.N
    Threads.@threads for k = 1:N-1
        if solver.control_integration == :foh
            res.fx[k], res.fu[k], res.fv[k] = solver.Fd(res.X[k], res.U[k], res.U[k+1])
            res.Ac[k], res.Bc[k] = solver.Fc(res.X[k], res.U[k])
        else
            res.fx[k], res.fu[k] = solver.Fd(res.X[k], res.U[k])
        end
        solver.c_jacobian(res.Cx[k], res.Cu[k], res.X[k],res.U[k])
    end

    if solver.control_integration == :foh
        res.Ac[N], res.Bc[N] = solver.Fc(res.X[N], res.U[N])
        solver.c_jacobian(res.Cx[N], res.Cu[N], res.X[N],res.U[N])
    end

    solver.c_jacobian(res.Cx_N, res.X[N])
    return nothing
end

r1 = copy(results_pendulum)
r2 = copy(results_pendulum)
solver = solver_pendulum

println("Jacobians")
@btime update_jacobians!(r1,solver)
@btime calculate_jacobians_thread!(r2,solver)

println("Derivatives")
@btime calculate_derivatives!(r1,solver,r1.X,r1.U)
@btime calculate_derivatives_thread!(r2,solver,r2.X,r2.U)

println("Midpoints")
@btime calculate_midpoints!(r1,solver,r1.X,r1.U)
@btime calculate_midpoints_thread!(r2,solver,r2.X,r2.U)

println("Cost")
@btime _cost(solver,r1,r1.X,r1.U)
@btime _cost_thread(solver,r1,r1.X,r1.U)

println("Cost (constraints)")
@btime cost_constraints(solver,r1)
@btime cost_constraints_thread(solver,r1)

# profiler
using Profile
using Juno
@profiler results_pendulum, stats_pendulum = solve(solver_pendulum,U_pendulum)
#
Juno.profiletree()
Juno.profiler()

# @time results_pendulum, stats_pendulum = solve(solver_pendulum,U_pendulum)
