using iLQR
using Plots

## double pendulum 
# dp = iLQR.Model("doublependulum.urdf")
# Q = 1e-4*eye(dp.n)
# R = 1e-4*eye(dp.m)
# Qf = 250.0*eye(dp.n)
# tf = 5.0
# x0 = [0.;0.;0.;0.]
# xf = [pi;0.;0.;0.]
# dt = 0.1
# obj = iLQR.Objective(Q,R,Qf,tf,x0,xf)
# solver = iLQR.Solver(dp, obj, dt)
# P = plot(linspace(0,tf,size(X,2)),X[1,:],title="Double Pendulum",label="\Theta")
# P = plot!(linspace(0,tf,size(X,2)),X[2,:],ylabel="State",label="\dot{\theta}")
# display(P)
# X, U, = @time iLQR.solve(solver)

## parallel park
function fc(x,u)
    return [u[1]*cos(x[3]); u[1]*sin(x[3]); u[2]]
end

n = 3 # number of states
m = 2 # number of controls

car = iLQR.Model(fc,n,m)

# initial and goal states
x0 = [0.;0.;0.]
xf = [0.;1.;0.]

# costs
Q = 0.001*eye(car.n)
Qf = 100.0*eye(car.n)
R = 0.001*eye(car.m)

# simulation
tf = 5.0
dt = 0.01

obj = iLQR.Objective(Q,R,Qf,tf,x0,xf)
solver = iLQR.Solver(car,obj,dt);
X, U, = @time iLQR.solve(solver)

P = plot((x0[1],x0[2]),marker=(:circle,"red"),label="x0")
P = plot!((xf[1],xf[2]),marker=(:circle,"green"),label="xf")
P = plot!(X[1,:],X[2,:],title="Dubins Car Parallel Park",label="traj.",color="blue",xlim=(-2,2),ylim=(-2,2))
display(P)
