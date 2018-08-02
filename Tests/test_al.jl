include("../iLQR.jl")
include("../dynamics.jl")
using iLQR
using Dynamics
using Plots
using BenchmarkTools


solver = iLQR.Solver(Dynamics.pendulum...,dt=0.1)
solver.opts.verbose = true

# Regular dynamics
U = ones(solver.model.m, solver.N-1)
@time results = iLQR.solve(solver,U)

# In place dynamics
opt = iLQR.SolverOptions()
opt.inplace_dynamics = true
obj_uncon = Dynamics.pendulum[2]
obj = iLQR.ConstrainedObjective(obj_uncon, u_min=-2, u_max=2)
model! = iLQR.Model(Dynamics.pendulum_dynamics!,2,1)
obj_uncon = Dynamics.pendulum[2]


# Unconstrained problem
solver = iLQR.Solver(model!, obj_uncon,dt=0.1,opts=opt)
@time results_c = iLQR.solve(solver, U)
solver.opts.verbose = false
@btime iLQR.solve(solver, U)



# Constrained problem
obj = iLQR.ConstrainedObjective(obj_uncon, u_min=-2, u_max=2)
solver! = iLQR.Solver(model!,obj,dt=0.1,opts=opt)
solver!.obj.Qf .= eye(2)*100.0
solver!.opts.verbose = true
# @enter iLQR.solve_al(solver!,U)
@time results_c = iLQR.solve(solver!,U)
solver!.opts.verbose = false
# @btime xc, uc = iLQR.solve(solver!,U)
# @profile xc, uc = iLQR.solve(solver!,U)

solver = iLQR.Solver(model!, obj_uncon, dt=0.1, opts=opt)
@time x1,u1 = iLQR.solve(solver,U)
@btime x1,u1 = iLQR.solve(solver,U)


plot(xc',label=["pos (constrained)", "vel (constrained)"], color=[:red :blue])
plot!(x1',label=["pos (constrained)" "vel (constrained)"], color=[:red :blue], width=2)
plot(uc',label="constrained")
plot!(u1',label="unconstrained")

### infeasible start
opts = iLQR.SolverOptions()
opts.inplace_dynamics = true
obj_uncon = Dynamics.pendulum[2]
model! = iLQR.Model(Dynamics.pendulum_dynamics!,2,1)

obj = iLQR.ConstrainedObjective(obj_uncon, u_min=-2, u_max=2)
solver! = iLQR.Solver(model!,obj,dt=0.1,opts=opts)
solver!.opts.verbose = true
U = ones(solver!.model.m, solver!.N-1)
U_ = ones(solver!.model.m+solver!.model.n,solver!.N-1)
X = ones(solver!.model.n, solver!.N)

results = iLQR.solve_al(solver!,X,U)
results2 = iLQR.solve_al(solver!,U)
#@time results_c = iLQR.solve(solver!,U)
###

# c_fun, constraint_jacobian = iLQR.generate_constraint_functions(obj,infeasible=true)
# c_fun(7*ones(2,1),3.5*ones(3,1))
# constraint_jacobian(7*ones(2,1),3.5*ones(3,1))
# Xb, b= iLQR.infeasible_bias(solver!,X,U)
# iLQR.cost(solver!,X,U_,infeasible=true)
#
# plot(results.X')
# println(results.X[:,end])
# plot(results.U')
# iLQR.rollout!(solver!,Xb,U_,infeasible=true)
#
# results.C
