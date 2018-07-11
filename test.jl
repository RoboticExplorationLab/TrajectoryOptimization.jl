include("iLQR.jl")
include("dynamics.jl")
using iLQR

# # initial and goal states
# x0 = [0.;0.;0.;0.]
# xf = [pi;0.;0.;0.]
#
# # costs
# Q = 0.0001*eye(4)
# Qf = 250.0*eye(4)
# R = 0.0001*eye(2)
#
# # simulation
# tf = 5.0
# dt = 0.1
#
# obj = Objective(Q,R,Qf,tf,x0,xf)
# println("type: $(typeof(doublependulum))")
# solver = Solver(doublependulum,obj,dt);
#
# #U = 10.0*rand(solver.model.m,solver.N)
# #X_dp, U_dp = @time iLQR.solve(solver,U);



#P = plot(linspace(0,tf,solver.N),X_dp[1,:],title="Double Pendulum",label="theta")
#P = plot!(linspace(0,tf,solver.N),X_dp[2,:],ylabel="State",label="theta_dot")

## Simple Pendulum
n = 2 # number of states
m = 1 # number of controls

function fc(x,u)
    # continuous dynamics (as defined in https://github.com/HarvardAgileRoboticsLab/unscented-dynamic-programming/blob/master/pendulum_dynamics.m)
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    return [x[2]; (u - m*g*lc*sin(x[1]) - b*x[2])];
end

#initial and goal conditions
x0 = [0.; 0.]
xf = [pi; 0.] # (ie, swing up)

#costs
Q = 0.3*eye(n)
Qf = 30.0*eye(n)
R = 0.3*eye(m)

#simulation
dt = 0.1
tf = 5.0
pendulum = Model(fc,n,m)
obj = Objective(Q,R,Qf,tf,x0,xf)
println(obj.Q)
solver = Solver(pendulum, obj::Objective, dt::Float64)
