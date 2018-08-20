using TrajectoryOptimization
using Plots
using BenchmarkTools

## Simple Pendulum Tests
# 1. Unconstrained
# 2. Unconstrained (inplace dynamics)
# 3. Constrained controls (inplace dynamics)
# 4. Constrained controls and constrained homogenous states (inplace dynamics)
# 5. Constrained controls and constrained hetergeneous states (inplace dynamics)
# 6. Infeasible start with constrained controls and constrained hetergeneous states (inplace dynamics)
# 7. Randomized infeasible start constrained controls and constrained hetergeneous states (inplace dynamics)

n = 2 # number of pendulum states
m = 1 # number of pendulum controls
opts = SolverOptions()
# opts.inplace_dynamics = true
opts.square_root = false
opts.verbose = true
opts.cache = true
obj_uncon = Dynamics.pendulum[2]

# obj_uncon.Qf[:,:] = 30.0*eye(n)
# obj_uncon.Q[:,:] = 0.3*eye(n)
# obj_uncon.R[:,:] = 0.3*eye(m)
# obj_uncon.tf = 5.0

# obj_uncon.Qf[:,:] = 100.0*eye(n)
# obj_uncon.Q[:,:] = 0.01*eye(n)
# obj_uncon.R[:,:] = 0.01*eye(m)

model! = Model(Dynamics.pendulum_dynamics!,n,m) # inplace dynamics model

# # 1. Unconstrained
# solver1 = Solver(model!,obj_uncon,integration=:rk3,dt=0.1)
# solver1.opts.verbose = true
# solver1.opts.cache = true
# solver1.opts.iterations = 250
# solver1.opts.eps = 1e-3
# U1 = zeros(m, solver1.N-1)
# @time results1 = solve_unconstrained(solver1,U1)
# println("Average time per iteration: $(sum(results1.time)/results1.termination_index)(s)")
# anim = @animate for i=1:5#size(results1.X,2)
#     x = cos(results1.X[1,i] - pi/2)
#     y = sin(results1.X[1,i] - pi/2)
#     plot([0,x],[0,y],xlims=(-1.5,1.5),ylims=(-1.5,1.5),color="black",size=(200,200))#,label="",title="Pendulum")
# end

#pendulum_animation(results1,fps=5)
# results1.result[1].X
#
# res = UnconstrainedResults(n,m,solver1.N)
# res.U[:,:] = U1[:,:]
# rollout!(res,solver1)
# res.X
# solver1.F(res.X[:,2],res.U[2])
#@btime results1 = solve(solver1,U1)

# plot(results1.X',title="Pendulum (1. Unconstrained)",ylabel="x(t)")
# results1.X[:,end]
# println("Average time per iteration: $(sum(results1.time)/results1.termination_index)(s)")
# plot(results1.cost[1:results1.termination_index-1],title="Cost",color="red")


# # 2. Unconstrained (inplace dynamics)
# solver2! = Solver(model!, obj_uncon,dt=0.1,opts=opts)
# U2 = ones(m,solver2!.N-1)
# @time results2 = solve(solver2!, U2)
# #@btime results2 = iLQR.solve(solver2!, U2)
# plot(results2.X',title="Pendulum (2. Unconstrained (inplace dynamics))",ylabel="x(t)")
#
#
# # 3. Constrained control (inplace dynamics)
# u_min3 = -2
# u_max3 = 2
# obj3 = ConstrainedObjective(obj_uncon, u_min=u_min3, u_max=u_max3)
# solver3! = Solver(model!,obj3,dt=0.1,opts=opts)
# U3 = ones(m,solver3!.N-1)
# @time results3 = solve_al(solver3!,U3)
# #plot(results3.X',title="Pendulum (3. Constrained (inplace dynamics))",ylabel="x(t)")
# #plot(results3.U',title="Pendulum (3. Constrained (inplace dynamics))",ylabel="u(t)")
#
# # 4. Constrained control and homogeneous states (inplace dynamics)
# u_min4 = -2
# u_max4 = 2
# x_min4 = -2
# x_max4 = 10
# obj4 = ConstrainedObjective(obj_uncon, u_min=u_min4, u_max=u_max4, x_min=x_min4, x_max=x_max4)
# solver4! = Solver(model!,obj4,dt=0.1,opts=opts)
# U4 = ones(m,solver4!.N-1)
# @time results4 = solve_al(solver4!,U4)
# plot(results4.X',title="Pendulum (4. Constrained control and states (inplace dynamics))",ylabel="x(t)")
# plot(results4.U',title="Pendulum (4. Constrained control and states (inplace dynamics))",ylabel="u(t)")
#
# # 5. Constrained control and heterogeneous states (inplace dynamics)
# u_min5 = -2
# u_max5 = 2
# x_min5 = [-1;-2]
# x_max5 = [10; 7]
# obj5 = ConstrainedObjective(obj_uncon, u_min=u_min5, u_max=u_max5, x_min=x_min5, x_max=x_max5)
# solver5! = Solver(model!,obj5,dt=0.1,opts=opts)
# U5 = ones(m,solver5!.N-1)
# @time results5 = solve_al(solver5!,U5)
# #plot(results5.X',title="Pendulum (5. Constrained control and states (inplace dynamics))",ylabel="x(t)")
# #plot(results5.U',title="Pendulum (5. Constrained control and states (inplace dynamics))",ylabel="u(t)")
#
# 6. Infeasible start with constrained control and heterogeneous states (inplace dynamics)
opts.cache=true
u_min6 = -3
u_max6 = 3
x_min6 = [-Inf;-Inf]
x_max6 = [Inf; Inf]
obj6 = ConstrainedObjective(obj_uncon, u_min=u_min6, u_max=u_max6, x_min=x_min6, x_max=x_max6)
solver6! = Solver(model!,obj6,dt=0.1,opts=opts)
U6 = ones(m,solver6!.N)
#X06 = ones(n,solver6!.N)
X_interp = line_trajectory(solver6!.obj.x0,solver6!.obj.xf,solver6!.N)
@time results6 = solve(solver6!,X_interp,U6)
plot(results6.X',title="Pendulum (6. Infeasible start with constrained control and states (inplace dynamics))",ylabel="x(t)")
plot(results6.U',title="Pendulum (6. Infeasible start with constrained control and states (inplace dynamics))",ylabel="u(t)")

results6.U
plot(log.(results6.cost))
results6.iter_type
println(plot_cost(results6))

index_outerloop = findall(x -> x == 1, results6.iter_type)
results6.cost[index_outerloop-1]
results6.termination_index
