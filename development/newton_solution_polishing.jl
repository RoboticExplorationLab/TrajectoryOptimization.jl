using Test
using BenchmarkTools
using Plots
using SparseArrays

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-6
opts.constraint_tolerance = 1e-4
opts.square_root = true
opts.active_constraint_tolerance = 0.0
opts.outer_loop_update_type = :default
opts.live_plotting = false
###
model, obj = TrajectoryOptimization.Dynamics.double_integrator
u_min = -0.2
u_max = 0.2
obj_con = TrajectoryOptimization.ConstrainedObjective(obj, tf=5.0,use_xf_equality_constraint=true, u_min=u_min, u_max=u_max)#, x_min=x_min, x_max=x_max)
###

###
model, obj = TrajectoryOptimization.Dynamics.pendulum
obj_con = Dynamics.pendulum_constrained[2]
obj_con.u_min[1] = -1
###

###
model, obj_con = TrajectoryOptimization.Dynamics.dubinscar_parallelpark
###

solver = TrajectoryOptimization.Solver(model,obj_con,integration=:rk4,N=100,opts=opts)
U = rand(solver.model.m, solver.N)

results, stats = TrajectoryOptimization.solve(solver,U)
@assert max_violation(results) < opts.constraint_tolerance
# plot(results.X,title="Block push to origin",xlabel="time step",ylabel="state",label=["pos.";"vel."])
# plot(results.U,title="Block push to origin",xlabel="time step",ylabel="control")
stats["cost"][end]
stats["c_max"][end]


x_pre = vcat(results.X...)
λ_pre = vcat(results.λ...)
c_pre = vcat(results.C...)
c_pre[321]
_alpha = 1.0
results_new = copy(results)
newton_results = NewtonResults(solver)
update_newton_results!(newton_results,results_new,solver)
J_prev = cost_newton(results_new,newton_results,solver)
max_violation(results_new)

newton_results.g[321]
newton_results.λ[321]
newton_results.active_set[321]
newton_results.active_set_ineq[321]
active_set_criteria(solver,newton_results.g[321],newton_results.λ[321],10)
findall(x->x==0.0,newton_results.λ[newton_results.active_set_ineq])

for i = 1:403
    if newton_results.λ[i] == 0.0 && newton_results.active_set_ineq[i] == true
        println(i)
        error("HI")
    end
end
rank(Diagonal(newton_results.λ .*newton_results.active_set_ineq))
rank(Array(newton_results.F))
rank(Array(newton_results.A))

sum(newton_results.active_set)
sum(newton_results.active_set_ineq)

newton_step!(results_new,newton_results,solver,1.0)
J = cost_newton(results_new,newton_results,solver)
max_violation(results_new)
sum(vcat(results.active_set...) - vcat(results_new.active_set...))

x_post = vcat(results_new.X...)
λ_post = vcat(results_new.λ...)
c_post = vcat(results_new.C...)
norm(x_post .- x_pre)
norm(λ_post .- λ_pre)
norm(c_post .- c_pre)
λ_post[321]
λ_update_default!(results_new,solver)

results = copy(results_new)
update_constraints!(results_new,solver)
update_newton_results!(newton_results,results_new,solver)


if J <= J_prev + 0.01*newton_results.b'*_alpha*newton_results.δ
    results = copy(results_new)
    J_prev = copy(J)
    _alpha = 1.0;
else
    results_new = copy(results)
    _alpha /= 2.0;
end

newton_solve!(copy(results),solver)

# @benchmark newton_solve($results,$newton_results,$solver,$1.0)

# plot(results_newton.U,title="Pendulum",xlabel="time step",ylabel="control",label="Newton",legend=:bottomright)
# plot!(results.U,label="AuLa")

x_min = obj_con.x_min
x_max = obj_con.x_max
plt = plot(title="Parallel Park")#,aspect_ratio=:equal)
plot!(x_min[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
plot!(x_max[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_min[2]*ones(1000),color=:red,width=2,label="")
plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_max[2]*ones(1000),color=:red,width=2,label="")
plot_trajectory!(to_array(results_newton.X),width=2,color=:blue,label="Newton",legend=:bottomright)
plot_trajectory!(to_array(results.X),width=2,color=:green,label="AuLa",legend=:bottomright)


# Cost
cost(solver,results)
J_newton
# Final max constraint tolerance
max_violation(results)
max_violation(results_newton)
