using Test
using Plots

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = true

model, obj = TrajectoryOptimization.Dynamics.double_integrator
u_min = -1.0
u_max = 1.0
obj_con = TrajectoryOptimization.ConstrainedObjective(obj, u_min=u_min, u_max=u_max, use_terminal_state_equality_constraint=false)
obj_con.tf = 20.0
integrator = :rk3
dt = 0.05
solver = TrajectoryOptimization.Solver(model,obj_con,integration=integrator,dt=dt,opts=opts)
get_num_constraints(solver)
U = zeros(solver.model.m, solver.N)
results, stats = TrajectoryOptimization.solve(solver,U)

plot(to_array(results.U)',label="")
max_violation(results)
maximum(abs.(to_array(results.U)))

function λ_second_order_update(res::ConstrainedIterResults, solver::Solver, bp::BackwardPass)
    for k = 1:N-1
        Q = [bp.Qxx[k] bp.Qux[k]; bp.Qux[k]' bp.Quu[k]]
        Cz = [results.Cx[k] results.Cu[k]]
    end
end
