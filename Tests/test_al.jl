include("../iLQR.jl")
include("../dynamics.jl")
using iLQR
using Dynamics
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
opts = iLQR.SolverOptions()
opts.inplace_dynamics = true
obj_uncon = Dynamics.pendulum[2]
model! = iLQR.Model(Dynamics.pendulum_dynamics!,n,m) # inplace dynamics model

# 1. Unconstrained
solver1 = iLQR.Solver(Dynamics.pendulum...,dt=0.1)
solver1.opts.verbose = true
U1 = ones(m, solver1.N-1)
@time results1 = iLQR.solve(solver1,U1)
#@btime results1 = iLQR.solve(solver1,U1)
#plot(results1.X',title="Pendulum (1. Unconstrained)",ylabel="x(t)")

# 2. Unconstrained (inplace dynamics)
solver2! = iLQR.Solver(model!, obj_uncon,dt=0.1,opts=opts)
U2 = ones(m,solver2!.N-1)
@time results2 = iLQR.solve(solver2!, U2)
#@btime results2 = iLQR.solve(solver2!, U2)
#plot(results2.X',title="Pendulum (2. Unconstrained (inplace dynamics))",ylabel="x(t)")


# 3. Constrained control (inplace dynamics)
u_min3 = -2
u_max3 = 2
obj3 = iLQR.ConstrainedObjective(obj_uncon, u_min=u_min3, u_max=u_max3)
solver3! = iLQR.Solver(model!,obj3,dt=0.1,opts=opts)
U3 = ones(m,solver3!.N-1)
@time results3 = iLQR.solve_al(solver3!,U3)
#plot(results3.X',title="Pendulum (3. Constrained (inplace dynamics))",ylabel="x(t)")
#plot(results3.U',title="Pendulum (3. Constrained (inplace dynamics))",ylabel="u(t)")

# 4. Constrained control and homogeneous states (inplace dynamics)
u_min4 = -2
u_max4 = 2
x_min4 = -2
x_max4 = 10
obj4 = iLQR.ConstrainedObjective(obj_uncon, u_min=u_min4, u_max=u_max4, x_min=x_min4, x_max=x_max4)
solver4! = iLQR.Solver(model!,obj4,dt=0.1,opts=opts)
U4 = ones(m,solver4!.N-1)
@time results4 = iLQR.solve_al(solver4!,U4)
#plot(results4.X',title="Pendulum (4. Constrained control and states (inplace dynamics))",ylabel="x(t)")
#plot(results4.U',title="Pendulum (4. Constrained control and states (inplace dynamics))",ylabel="u(t)")

# 5. Constrained control and heterogeneous states (inplace dynamics)
u_min5 = -2
u_max5 = 2
x_min5 = [-1;-2]
x_max5 = [10; 7]
obj5 = iLQR.ConstrainedObjective(obj_uncon, u_min=u_min5, u_max=u_max5, x_min=x_min5, x_max=x_max5)
solver5! = iLQR.Solver(model!,obj5,dt=0.1,opts=opts)
U5 = ones(m,solver5!.N-1)
@time results5 = iLQR.solve_al(solver5!,U5)
#plot(results5.X',title="Pendulum (5. Constrained control and states (inplace dynamics))",ylabel="x(t)")
#plot(results5.U',title="Pendulum (5. Constrained control and states (inplace dynamics))",ylabel="u(t)")

# 6. Infeasible start with constrained control and heterogeneous states (inplace dynamics)
u_min6 = -2
u_max6 = 2
x_min6 = [-1;-2]
x_max6 = [10; 7]
obj6 = iLQR.ConstrainedObjective(obj_uncon, u_min=u_min6, u_max=u_max6, x_min=x_min6, x_max=x_max6)
solver6! = iLQR.Solver(model!,obj6,dt=0.1,opts=opts)
U6 = ones(m,solver6!.N-1)
X06 = ones(n,solver6!.N)
@time results6 = iLQR.solve_al(solver6!,X06, U6)
plot(results6.X',title="Pendulum (6. Infeasible start with constrained control and states (inplace dynamics))",ylabel="x(t)")
plot(results6.U',title="Pendulum (6. Infeasible start with constrained control and states (inplace dynamics))",ylabel="u(t)")

Xb, b= iLQR.infeasible_bias(solver6!,X06,U6)
plot(b',label=["bias1","bias2"])
plot!(results6.U[2:3,:]')
