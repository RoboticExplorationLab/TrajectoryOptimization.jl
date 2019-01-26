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
solver.opts.restype = Matrix
solver_new = Solver(model,obj_con,N=N)
# solver_new.opts.restype = TrajectoryVariable
solver.opts.verbose = true
solver.state.infeasible
n,m = get_sizes(solver)

X0 = Array{Float64,2}(undef,0,0)
U0 = ones(m,N)

solver.opts.constrained
solver_new.opts.constrained

#****************************#
#       INITIALIZATION       #
#****************************#
mbar,mm = get_num_controls(solver)
results = init_results(solver, X0, U0)
results_new = init_results(solver_new, X0, U0)
results.ρ[1] = 0
results_new.ρ[1] = 0
X,U = results.X, results.U
X_,U_ = results.X_, results.U_

bp = TrajectoryOptimization.BackwardPass(n,mm,N)
#****************************#
#           SOLVER           #
#****************************#
## Initial rollout
if !solver.state.infeasible
    X[1] = solver.obj.x0
    flag = rollout!(results,solver) # rollout new state trajectoy
    rollout!(results_new,solver_new)

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

results.X == results_new.X.x
results.μ == results_new.μ.x



J_prev = cost(solver, results)
TrajectoryOptimization.update_jacobians!(results, solver)
Δv = backwardpass!(results, solver)
J = forwardpass!(results, solver, Δv)
forwardpass!(results_new, solver_new, Δv)

b1 = @benchmark backwardpass!($results, $solver, $bp)
b2 = @benchmark backwardpass!($results_new, $solver_new, $bp)
judge(median(b2),median(b1))

b1 = @benchmark forwardpass!($results, $solver, $Δv)
b2 = @benchmark forwardpass!($results_new, $solver_new, $Δv)
judge(median(b2),median(b1))


results.X .= deepcopy(results.X_)
results.U .= deepcopy(results.U_)
copyto!(results_new.X,results_new.X_)
copyto!(results_new.U,results_new.U_)


b2 = @benchmark copyto!(results_new.X,results_new.X_)
b1 = @benchmark results.X .= deepcopy(results.X_)
judge(median(b2),median(b1))

outer_loop_update(results,solver)
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
