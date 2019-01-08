using Juno
using Profile
using BenchmarkTools

# Set up
N = 51
model, obj = Dynamics.dubinscar
n,m = model.n, model.m

x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]

obj_con = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max)

solver = Solver(model,obj_con,N=N)
solver.opts.verbose = true
solver.state.infeasible
n,m = get_sizes(solver)

X0 = Array{Float64,2}(undef,0,0)
U0 = ones(m,N)

# Profile.init(n=10^7,delay=0.0001)

#****************************#
#       INITIALIZATION       #
#****************************#
results = init_results(solver, X0, U0)
X,U = results.X, results.U
X_,U_ = results.X_, results.U_

#****************************#
#           SOLVER           #
#****************************#
## Initial rollout
if !solver.state.infeasible
    X[1] = solver.obj.x0
    flag = rollout!(results,solver) # rollout new state trajectoy

    if !flag
        @info "Bad initial control sequence, setting initial control to zero"
        results.U .= zeros(mm,N)
        rollout!(results,solver)
    end
end

if solver.state.infeasible
    if solver.control_integration == :foh
        calculate_derivatives!(results, solver, results.X, results.U)
        calculate_midpoints!(results, solver, results.X, results.U)
    end
    update_constraints!(results,solver,results.X,results.U)
end
p = plot()
plot_trajectory!(results)
# plot(to_array(results.X)')

J_prev = cost(solver, results)
TrajectoryOptimization.update_jacobians!(results, solver)
Δv = backwardpass!(results, solver)
J = forwardpass!(results, solver, Δv)

Profile.init(delay=1e-4)
Profile.clear()
@profile backwardpass!(results, solver)
Juno.profiler()
Profile.print()

Profile.clear()
@profile forwardpass!(results, solver, Δv)
Juno.profiler()
Profile.print()


X .= deepcopy(X_)
U .= deepcopy(U_)

plot_trajectory!(results)
λ_update!(results, solver)


# Warm start
λ = deepcopy(results.λ)
push!(λ, results.λN)
U0_warm = to_array(results.U)
res_warm = init_results(solver, X0, U0_warm, λ=λ)
rollout!(res_warm,solver)

J_prev = cost(solver, res_warm)
update_jacobians!(res_warm, solver)
Δv = backwardpass!(res_warm, solver)
J = forwardpass!(res_warm, solver, Δv)
