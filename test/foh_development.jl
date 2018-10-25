# The purpose of this script is to validate first order hold derivation (including minimum time)
using ForwardDiff
using LinearAlgebra
using Test
model, obj = TrajectoryOptimization.Dynamics.dubinscar
n = model.n
m = model.m
dt = 1.0
h = sqrt(dt)
solver = Solver(model,obj,integration=:rk3_foh,dt=dt)
results = UnconstrainedVectorResults(model.n,model.m,solver.N)
U = ones(solver.model.m,solver.N)
copyto!(results.U,U)
rollout!(results,solver)
calculate_jacobians!(results,solver)

# Results from backward pass
k = 2
Ac1, Bc1 = results.fcx[k], results.fcu[k]
Ac2, Bc2 = results.fcx[k+1], results.fcu[k+1]
Q = obj.Q
R = getR(solver)
xf = obj.xf
X = results.X
U = results.U
xm = results.xm[k]
um = (results.U[k] + results.U[k+1])/2.

Lx = dt/6*Q*(X[k] - xf) + 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(xm - xf)
Lu = dt/6*R*U[k] + 4*dt/6*((dt/8*Bc1)'*Q*(xm - xf) + 0.5*R*um)
Ly = dt/6*Q*(X[k+1] - xf) + 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(xm - xf)
Lv = dt/6*R*U[k+1] + 4*dt/6*((-dt/8*Bc2)'*Q*(xm - xf) + 0.5*R*um)

Lxx = dt/6.0*Q + 4.0*dt/6.0*(I/2.0 + dt/8.0*Ac1)'*Q*(I/2.0 + dt/8.0*Ac1)
Luu = dt/6*R + 4*dt/6*((dt/8*Bc1)'*Q*(dt/8*Bc1) + 0.5*R*0.5)
Lyy = dt/6*Q + 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(I/2 - dt/8*Ac2)
Lvv = dt/6*R + 4*dt/6*((-dt/8*Bc2)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)

Lxu = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(dt/8*Bc1)
Lxy = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(I/2 - dt/8*Ac2)
Lxv = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(-dt/8*Bc2)
Luy = 4*dt/6*(dt/8*Bc1)'*Q*(I/2 - dt/8*Ac2)
Luv_ = 4*dt/6*((dt/8*Bc1)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)  # note the name change; workspace conflict
Lyv = 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(-dt/8*Bc2)

# test functions
function el(x,u)
    0.5*(x - xf)'*Q*(x - xf) + 0.5*u'*R*u
end

function el(s)
    x = s[1:n]
    u = s[n+1:n+m]
    el(x,u)
end

function x_midpoint(x,y,xdot,ydot,h)
    0.5*x + 0.5*y + (h^2)/8*xdot - (h^2)/8*ydot
end

function u_midpoint(u,v)
    0.5*u + 0.5*v
end

function fc_(x,u)
    xdot = zero(x)
    model.f(xdot,x,u)
    xdot
end

function fc_(s)
    x = s[1:n]
    u = s[n+1:n+m]
    fc_(x,u)
end

function stage_cost(x,y,u,v,h)
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    (h^2)/6*(el(x,u) + 4*el(xm,um) + el(y,v))
end

function stage_cost(s)
    x = s[1:n]
    u = s[n+1:n+m]
    y = s[n+m+1:n+m+n]
    v = s[n+m+n+1:n+m+n+m]
    h = s[n+m+n+m+1]
    stage_cost(x,y,u,v,h)
end

x = results.X[k]
y = results.X[k+1]
u = results.U[k]
v = results.U[k+1]

s = [x;u;y;v;h]
stage_cost(x,y,u,v,h)
stage_cost(s)

# Get ForwardDiff gradients and Jacobians
L_gradient = ForwardDiff.gradient(stage_cost,s)

@test isapprox(L_gradient[1:n],Lx)
@test isapprox(L_gradient[n+1:n+m],Lu)
@test isapprox(L_gradient[n+m+1:n+m+n],Ly)
@test isapprox(L_gradient[n+m+n+1:n+m+n+m],Lv)

# NOTE: Because we neglect second-orer derivatives of the dynamics (continous or discrete) we can't simply call ForwardDiff's Hessian and get results that match the equations we use in backwardpass_foh, we need to make functions that match the gradients first, then take the jacobians of the gradients
function Lx_func(x,u,y,v,h)
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    (h^2)/6*Q*(x - xf) + 4*(h^2)/6*(I/2 + (h^2)/8*Ac1)'*Q*(xm - xf)
end
function Lx_func(s)
    x = s[1:n]
    u = s[n+1:n+m]
    y = s[n+m+1:n+m+n]
    v = s[n+m+n+1:n+m+n+m]
    h = s[n+m+n+m+1]
    Lx_func(x,u,y,v,h)
end

function Lu_func(x,u,y,v,h)
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    (h^2)/6*R*u + 4*(h^2)/6*(((h^2)/8*Bc1)'*Q*(xm - xf) + 0.5*R*um)
end
function Lu_func(s)
    x = s[1:n]
    u = s[n+1:n+m]
    y = s[n+m+1:n+m+n]
    v = s[n+m+n+1:n+m+n+m]
    h = s[n+m+n+m+1]
    Lu_func(x,u,y,v,h)
end

function Ly_func(x,u,y,v,h)
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    Ly = (h^2)/6*Q*(y - xf) + 4*(h^2)/6*(I/2 - (h^2)/8*Ac2)'*Q*(xm - xf)
end
function Ly_func(s)
    x = s[1:n]
    u = s[n+1:n+m]
    y = s[n+m+1:n+m+n]
    v = s[n+m+n+1:n+m+n+m]
    h = s[n+m+n+m+1]
    Ly_func(x,u,y,v,h)
end

function Lv_func(x,u,y,v,h)
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    Lv = (h^2)/6*R*v + 4*(h^2)/6*((-(h^2)/8*Bc2)'*Q*(xm - xf) + 0.5*R*um)
end
function Lv_func(s)
    x = s[1:n]
    u = s[n+1:n+m]
    y = s[n+m+1:n+m+n]
    v = s[n+m+n+1:n+m+n+m]
    h = s[n+m+n+m+1]
    Lv_func(x,u,y,v,h)
end

@test isapprox(ForwardDiff.jacobian(Lx_func,s)[1:n,1:n],Lxx)
@test isapprox(ForwardDiff.jacobian(Lu_func,s)[1:m,n+1:n+m],Luu)
@test isapprox(ForwardDiff.jacobian(Ly_func,s)[1:n,n+m+1:n+m+n],Lyy)
@test isapprox(ForwardDiff.jacobian(Lv_func,s)[1:m,n+m+n+1:n+m+n+m],Lvv)

@test isapprox(ForwardDiff.jacobian(Lx_func,s)[1:n,n+1:n+m],Lxu)
@test isapprox(ForwardDiff.jacobian(Lx_func,s)[1:n,n+m+1:n+m+n],Lxy)
@test isapprox(ForwardDiff.jacobian(Lx_func,s)[1:n,n+m+n+1:n+m+n+m],Lxv)
@test isapprox(ForwardDiff.jacobian(Lu_func,s)[1:m,n+m+1:n+m+n],Luy)
@test isapprox(ForwardDiff.jacobian(Lu_func,s)[1:m,n+m+n+1:n+m+n+m],Luv_)
@test isapprox(ForwardDiff.jacobian(Ly_func,s)[1:n,n+m+n+1:n+m+n+m],Lyv)
