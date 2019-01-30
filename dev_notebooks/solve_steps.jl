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

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]

obj_con = ConstrainedObjective(obj,x_min=x_min,x_max=x_max)

solver = Solver(model,obj_con,N=N)
solver.opts.verbose = true
solver.state.infeasible
n,m = get_sizes(solver)

solver2 = Solver(model,obj_con,N=N)
solver2.opts.al_type = :algencan

X0 = Array{Float64,2}(undef,0,0)
U0 = ones(m,N)

logger = default_logger(solver)
global_logger(logger)

#****************************#
#       INITIALIZATION       #
#****************************#
mbar,mm = get_num_controls(solver)
results = init_results(solver, X0, U0)
results2 = init_results(solver2, X0, U0)
results.ρ[1] = 0
results2.ρ[1] = 0
X,U = results.X, results.U
X_,U_ = results.X_, results.U_

#****************************#
#           SOLVER           #
#****************************#
## Initial rollout
if !solver.state.infeasible
    X[1] = solver.obj.x0
    flag = rollout!(results,solver) # rollout new state trajectoy
    rollout!(results2,solver2)

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
    update_constraints!(results_new,solver_new,results.X,results.U)
end
# p = plot()
# plot_trajectory!(results)
# plot(to_array(results.X)')

# results.X == results_new.X.x
# results.μ == results_new.μ.x



J_prev = cost(solver, results)
TrajectoryOptimization.update_jacobians!(results, solver)
Δv = backwardpass!(results, solver)
J = forwardpass!(results, solver, Δv, J_prev)
c_max = max_violation(results)
results.X .= deepcopy(results.X_)
results.U .= deepcopy(results.U_)

print_header(logger, InnerLoop)
J_prev = cost(solver2, results2)
update_jacobians!(results2, solver2)
Δv = _backwardpass_new!(results2, solver2)
J = forwardpass!(results2, solver2, Δv, J_prev)
c_max = max_violation(results2)
print_row(logger, InnerLoop)

copyto!(results2.X,results2.X_)
copyto!(results2.U,results2.U_)


outer_loop_update(results2, solver2)


#
# b2 = @benchmark copyto!(results_new.X,results_new.X_)
# b1 = @benchmark results.X .= deepcopy(results.X_)
# judge(median(b2),median(b1))

outer_loop_update(results,solver)
cost(solver,results)
max_violation(results)
update_constraints!(results, solver)

outer_loop_update(results_new,solver_new)
update_constraints!(results_new, solver_new)

max_violation(results)
max_violation(results_new)


b1 = @benchmark outer_loop_update($results,$solver)
b2 = @benchmark outer_loop_update($results_new,$solver_new)
judge(median(b2),median(b1))

b1 = @benchmark update_constraints!($results,$solver)
b2 = @benchmark update_constraints!($results_new,$solver_new)
judge(median(b2),median(b1))

b1 = @benchmark max_violation($results)
b2 = @benchmark max_violation($results_new)
judge(median(b2),median(b1))



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
