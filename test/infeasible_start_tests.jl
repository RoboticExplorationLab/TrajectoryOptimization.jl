using TrajectoryOptimization
using Plots
using BenchmarkTools
using Base.Test
#using Juno

##### Set up pendulum for infeasible tests

n = 2 # number of pendulum states
m = 1 # number of pendulum controls
model! = Model(Dynamics.pendulum_dynamics!,n,m) # inplace dynamics model

opts = SolverOptions()
opts.square_root = false
opts.verbose = false
opts.cache=true
opts.c1=1e-4
opts.c2=2.0
opts.mu_al_update = 100.0
opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-3
opts.eps = 1e-5
opts.iterations_outerloop = 250
opts.iterations = 1000

# Constraints
u_min = -3
u_max = 3
x_min = [-10;-10]
x_max = [10; 10]
obj_uncon = Dynamics.pendulum[2]
obj_uncon.R[:] = [1e-2]
obj = ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)

solver = Solver(model!,obj,dt=0.1,opts=opts)
#####

# test linear interpolation for state trajectory
X_interp = line_trajectory(solver.obj.x0,solver.obj.xf,solver.N)
U = ones(solver.model.m,solver.N-1)
# @test norm(X_interp[:,1] - solver.obj.x0) < 1e-8
# @test norm(X_interp[:,end] - solver.obj.xf) < 1e-8
# @test all(X_interp[1,2:end-1] .<= max(solver.obj.x0[1],solver.obj.xf[1]))
# @test all(X_interp[2,2:end-1] .<= max(solver.obj.x0[2],solver.obj.xf[2]))

# test that additional augmented controls can achieve an infeasible state trajectory
# U0 = ones(m,solver.N-1)
# X0 = ones(n,solver.N)
# solver.obj.x0 = ones(n)
# ui = infeasible_controls(solver,X0,U0)
# results = ConstrainedResults(n,m+n,1,solver.N,1)
# results.U[:,:] = [U0;ui]
# rollout!(results,solver;infeasible=true)
#
# @test all(ui[1,:] .== ui[1,1]) # special case for state trajectory of all ones, control 1 should all be same
# @test all(ui[2,:] .== ui[2,1]) # special case for state trajectory of all ones, control 2 should all be same
# @test all(results.X .== X0) # rolled out trajectory should be equivalent to infeasible trajectory after applying augmented controls

@time results = solve_al(solver,X_interp,U)

plot(results.X',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="x(t)")
plot(results.U',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="u(t)")
println(results.X[:,end])
# trajectory_animation(results,filename="infeasible_start_state.gif",fps=5)
# trajectory_animation(results,traj="control",filename="infeasible_start_control.gif",fps=5)
idx = find(x->x==2,results.iter_type)
plot(results.result[end].X')

plot(results.result[idx[1]].U',color="green")
plot!(results.result[end].U',color="red")

norm(results.result[idx[1]].U-results.result[end].U)

tmp = ConstrainedResults(solver.model.n,solver.model.m,size(results.result[1].C,1),solver.N)
tmp.U[:,:] = results.result[idx[1]].U
tmp2 = ConstrainedResults(solver.model.n,solver.model.m,size(results.result[1].C,1),solver.N)
tmp2.U[:,:] = results.result[end].U

rollout!(tmp,solver)
plot(tmp.X')

rollout!(tmp2,solver)
plot!(tmp2.X')
