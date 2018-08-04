include("../iLQR.jl")
include("../dynamics.jl")
using iLQR
using Dynamics
using Plots
using BenchmarkTools

# solver = iLQR.Solver(Dynamics.pendulum...,dt=0.1)
# solver.opts.verbose = true
#
# U = ones(solver.model.m, solver.N-1)
# X = ones(solver.model.n,solver.N)
#
# opt = iLQR.SolverOptions()
# opt.inplace_dynamics = false
# opt.verbose = true
# opt.cache = true
#
# obj_uncon = Dynamics.pendulum[2]
# obj = iLQR.ConstrainedObjective(obj_uncon, u_min=-10., u_max=10.)
#
# model = iLQR.Model(Dynamics.pendulum_dynamics,2,1)
# solver = iLQR.Solver(model,obj,dt=0.1,opts=opt)
# solver.obj.Qf .*= 0
# @time xc, uc, x_cache, u_cache, i_count = iLQR.solve_infeasible(solver,X,U)

# anim = @animate for i=1:i_count-1
#
#     plt = plot(x_cache[:,:,i]',ylim=(-5,5),xlim=(0,solver.N),size=(200,200),label="",width=2,title="Pendulum state evolution infeasible start")
#     end
# gif(anim,"pendulum_infeasible_state_traj.gif",fps=5)
#
# anim = @animate for i=1:i_count-1
#
#     plt = plot(u_cache[1,:,i],ylim=(-5,5),size=(200,200),label="",width=2,title="Pendulum control evolution infeasible start")
#     end
# gif(anim,"pendulum_infeasible_control_traj.gif",fps=4)
### Inplace solve
solver = iLQR.Solver(Dynamics.pendulum...,dt=0.1)
solver.opts.verbose = true

U = ones(solver.model.m, solver.N-1)
X = ones(solver.model.n,solver.N)

opt = iLQR.SolverOptions()
opt.inplace_dynamics = true
opt.verbose = true
obj_uncon = Dynamics.pendulum[2]
# obj_uncon.Qf = eye(2)*1
obj = iLQR.ConstrainedObjective(obj_uncon, u_min=-10., u_max=10.)

model! = iLQR.Model(Dynamics.pendulum_dynamics!,2,1)
solver! = iLQR.Solver(model!,obj,dt=0.1,opts=opt)
solver!.obj.Qf .*= 0
@time xc, uc = iLQR.solve_infeasible(solver!,X,U)
