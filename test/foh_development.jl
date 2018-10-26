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

# Check that foh Hessians match ForwardDiff
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

# Gradients/Jacobians/Hessians for minimum time
dx = results.dx[k]
dy = results.dx[k+1]
dxm = 2*h/8*dx - 2*h/8*dy

L2h = 4/6*((h^2)*dxm'*Q*(xm - xf) + 2*h*el(xm,um))
L2hh = 4/6*(2/8*((h^3)*dxm'*Q*dx + 3*(h^2)*dx'*Q*xm) - 2/8*((h^3)*dxm'*Q*dy + 3*(h^2)*dy'*Q*xm) - 6*(h^2)/8*dx'*Q*xf + 6*(h^2)/8*dy'*Q*xf + 2*(h*dxm'*Q*(xm - xf) + el(xm,um)))
L2hu = 4*(h^2)/6*(2*(h^3)/8*(Bc1'*Q*xm + (h^2)/8*Bc1'*Q*dx) - 2*(h^3)/8*Bc1'*Q*xf - 2*(h^5)/64*Bc1'*Q*dy + 2*h*ℓ2u)

L2xh = 2/6*Q*(h*x + h*y + (h^3)/2*dx - (h^3)/2*dy) - 4/6*h*Q*xf + 1/12*Ac1'*Q*(2*(h^3)*x + 2*(h^3)*y + 6/8*(h^5)*dx -6/8*(h^5)*dy) - (h^3)/3*Ac1'*Q*xf
L2uh = 1/12*Bc1'*Q*(2*(h^3)*x + 2*(h^3)*y + 6/8*(h^5)*dx - 6/8*(h^5)*dy) - 1/3*(h^3)*Bc1'*Q*xf + 4/6*h*R*um
L2yh = 2/6*Q*(h*x + h*y + (h^3)/2*dx - (h^3)/2*dy) - 4/6*h*Q*xf - 1/12*Ac2'*Q*(2*(h^3)*x + 2*(h^3)*y + 6/8*(h^5)*dx -6/8*(h^5)*dy) + (h^3)/3*Ac2'*Q*xf
L2vh = -1/12*Bc2'*Q*(2*(h^3)*x + 2*(h^3)*y + 6/8*(h^5)*dx - 6/8*(h^5)*dy) + 1/3*(h^3)*Bc2'*Q*xf + 4/6*h*R*um

function L2_func(s)
    x = s[1:n]
    u = s[n+1:n+m]
    y = s[n+m+1:n+m+n]
    v = s[n+m+n+1:n+m+n+m]
    h = s[n+m+n+m+1]
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    4/6*(h^2)*el(xm,um)
end

function L2x_func(s)
    x = s[1:n]
    u = s[n+1:n+m]
    y = s[n+m+1:n+m+n]
    v = s[n+m+n+1:n+m+n+m]
    h = s[n+m+n+m+1]
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    4*(h^2)/6*(I/2 + (h^2)/8*Ac1)'*Q*(xm - xf)
end

function L2u_func(s)
    x = s[1:n]
    u = s[n+1:n+m]
    y = s[n+m+1:n+m+n]
    v = s[n+m+n+1:n+m+n+m]
    h = s[n+m+n+m+1]
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    4*(h^2)/6*(((h^2)/8*Bc1)'*Q*(xm - xf) + 0.5*R*um)
end

function L2y_func(s)
    x = s[1:n]
    u = s[n+1:n+m]
    y = s[n+m+1:n+m+n]
    v = s[n+m+n+1:n+m+n+m]
    h = s[n+m+n+m+1]
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    4*(h^2)/6*(I/2 - (h^2)/8*Ac2)'*Q*(xm - xf)
end

function L2v_func(s)
    x = s[1:n]
    u = s[n+1:n+m]
    y = s[n+m+1:n+m+n]
    v = s[n+m+n+1:n+m+n+m]
    h = s[n+m+n+m+1]
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    4*(h^2)/6*(-((h^2)/8*Bc2)'*Q*(xm - xf) + 0.5*R*um)
end

# Confirm that new partials match ForwardDiff
@test isapprox(ForwardDiff.gradient(L2_func,s)[end],L2h)
@test isapprox(ForwardDiff.hessian(L2_func,s)[end,end],L2hh)
@test isapprox(ForwardDiff.jacobian(L2x_func,s)[:,end],L2xh)
@test isapprox(ForwardDiff.jacobian(L2u_func,s)[:,end],L2uh)
@test isapprox(ForwardDiff.jacobian(L2y_func,s)[:,end],L2yh)
@test isapprox(ForwardDiff.jacobian(L2v_func,s)[:,end],L2vh)

## L(x,u,y,v) = L(x,u) + L(xm,um) + L(y,v) = L1 + L2 + L3
# Assembling ℓ(x,u) expansion
ℓ1 = el(x,u)
ℓ1x = Q*(x - xf)
ℓ1u = R*u
ℓ1y = zeros(n)
ℓ1v = zeros(m)

ℓ1xx = Q
ℓ1uu = R
ℓ1yy = zeros(n,n)
ℓ1vv = zeros(m,m)

ℓ1xu = zeros(n,m)
ℓ1xy = zeros(n,n)
ℓ1xv = zeros(n,m)
ℓ1uy = zeros(m,n)
ℓ1uv = zeros(m,m)
ℓ1yv = zeros(n,m)

# Assembling ℓ(xm,um) expansion
ℓ2 = el(xm,um)
ℓ2x = (I/2 + dt/8*Ac1)'*Q*(xm - xf)
ℓ2u = ((dt/8*Bc1)'*Q*(xm - xf) + 0.5*R*um)
ℓ2y = (I/2 - dt/8*Ac2)'*Q*(xm - xf)
ℓ2v = ((-dt/8*Bc2)'*Q*(xm - xf) + 0.5*R*um)

ℓ2xx = (I/2.0 + dt/8.0*Ac1)'*Q*(I/2.0 + dt/8.0*Ac1)
ℓ2uu = ((dt/8*Bc1)'*Q*(dt/8*Bc1) + 0.5*R*0.5)
ℓ2yy = (I/2 - dt/8*Ac2)'*Q*(I/2 - dt/8*Ac2)
ℓ2vv = ((-dt/8*Bc2)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)

ℓ2xu = (I/2 + dt/8*Ac1)'*Q*(dt/8*Bc1)
ℓ2xy = (I/2 + dt/8*Ac1)'*Q*(I/2 - dt/8*Ac2)
ℓ2xv = (I/2 + dt/8*Ac1)'*Q*(-dt/8*Bc2)
ℓ2uy = (dt/8*Bc1)'*Q*(I/2 - dt/8*Ac2)
ℓ2uv = ((dt/8*Bc1)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)  # note the name change; workspace conflict
ℓ2yv = (I/2 - dt/8*Ac2)'*Q*(-dt/8*Bc2)

# Confirm that partials of L2 still match ForwardDiff
@test isapprox(ForwardDiff.gradient(L2_func,s)[1:n],4*(h^2)/6*ℓ2x)
@test isapprox(ForwardDiff.gradient(L2_func,s)[n+1:n+m],4*(h^2)/6*ℓ2u)
@test isapprox(ForwardDiff.gradient(L2_func,s)[n+m+1:n+m+n],4*(h^2)/6*ℓ2y)
@test isapprox(ForwardDiff.gradient(L2_func,s)[n+m+n+1:n+m+n+m],4*(h^2)/6*ℓ2v)

@test isapprox(ForwardDiff.jacobian(L2x_func,s)[1:n,1:n],4*(h^2)/6*ℓ2xx)
@test isapprox(ForwardDiff.jacobian(L2u_func,s)[1:m,n+1:n+m],4*(h^2)/6*ℓ2uu)
@test isapprox(ForwardDiff.jacobian(L2y_func,s)[1:n,n+m+1:n+m+n],4*(h^2)/6*ℓ2yy)
@test isapprox(ForwardDiff.jacobian(L2v_func,s)[1:m,n+m+n+1:n+m+n+m],4*(h^2)/6*ℓ2vv)

@test isapprox(ForwardDiff.jacobian(L2x_func,s)[1:n,n+1:n+m],4*(h^2)/6*ℓ2xu)
@test isapprox(ForwardDiff.jacobian(L2x_func,s)[1:n,n+m+1:n+m+n],4*(h^2)/6*ℓ2xy)
@test isapprox(ForwardDiff.jacobian(L2x_func,s)[1:n,n+m+n+1:n+m+n+m],4*(h^2)/6*ℓ2xv)
@test isapprox(ForwardDiff.jacobian(L2u_func,s)[1:m,n+m+1:n+m+n],4*(h^2)/6*ℓ2uy)
@test isapprox(ForwardDiff.jacobian(L2u_func,s)[1:m,n+m+n+1:n+m+n+m],4*(h^2)/6*ℓ2uv)
@test isapprox(ForwardDiff.jacobian(L2y_func,s)[1:n,n+m+n+1:n+m+n+m],4*(h^2)/6*ℓ2yv)

# Assembling ℓ(y,v) expansion
ℓ3 = el(y,v)
ℓ3x = zeros(n)
ℓ3u = zeros(m)
ℓ3y = Q*(y - xf)
ℓ3v = R*v

ℓ3xx = zeros(n,n)
ℓ3uu = zeros(m,m)
ℓ3yy = Q
ℓ3vv = R

ℓ3xu = zeros(n,m)
ℓ3xy = zeros(n,n)
ℓ3xv = zeros(n,m)
ℓ3uy = zeros(m,n)
ℓ3uv = zeros(m,m)
ℓ3yv = zeros(n,m)

## Assemble action-value function Q(x,̂u,y,̂v) expansion where ̂u = [u;h], ̂v = [v;w]
Lx = dt/6*ℓ1x + 4*dt/6*ℓ2x + dt/6*ℓ3x
Lu = dt/6*ℓ1u + 4*dt/6*ℓ2u + dt/6*ℓ3u
Ly = dt/6*ℓ1y + 4*dt/6*ℓ2y + dt/6*ℓ3y
Lv = dt/6*ℓ1v + 4*dt/6*ℓ2v + dt/6*ℓ3v

Lxx = dt/6*ℓ1xx + 4*dt/6*ℓ2xx + dt/6*ℓ3xx
Luu = dt/6*ℓ1uu + 4*dt/6*ℓ2uu + dt/6*ℓ3uu
Lyy = dt/6*ℓ1yy + 4*dt/6*ℓ2yy + dt/6*ℓ3yy
Lvv = dt/6*ℓ1vv + 4*dt/6*ℓ2vv + dt/6*ℓ3vv

Lxu = dt/6*ℓ1xu + 4*dt/6*ℓ2xu + dt/6*ℓ3xu
Lxy = dt/6*ℓ1xy + 4*dt/6*ℓ2xy + dt/6*ℓ3xy
Lxv = dt/6*ℓ1xv + 4*dt/6*ℓ2xv + dt/6*ℓ3xv
Luy = dt/6*ℓ1uy + 4*dt/6*ℓ2uy + dt/6*ℓ3uy
Luv_ = dt/6*ℓ1uv + 4*dt/6*ℓ2uv + dt/6*ℓ3uv
Lyv = dt/6*ℓ1yv + 4*dt/6*ℓ2yv + dt/6*ℓ3yv

# Augmented expansions for minimum time
_Lx = Lx
_Lu = [Lu;2*h/6*ℓ1 + L2h + 2*h/6*ℓ3]
_Ly = Ly
_Lv = [Lv;0]

_Lxx = Lxx
_Luu = [Luu (2/6*h*ℓ1u + L2uh);(2/6*h*ℓ1u + L2hu)' (2/6*ℓ1 + L2hh + 2/6*ℓ3)]
_Lyy = Lyy
_Lvv = [Lvv zeros(m); zeros(m)' 0]

_Lxu = [Lxu (2/6*h*ℓ1x + L2xh)]
_Lxy = Lxy
_Lxv = [Lxv zeros(n)]
_Luy = [Luy' (L2yh + 2*h/6*ℓ3y)]'
_Luv = [Luv_' (L2vh + 2*h/6*ℓ3v); zeros(m)' 0]'
_Lyv = [Lyv zeros(n)]

###
function stage_cost_alt(s)
    x = s[1:n]
    u = s[n+1:n+m]
    h = s[n+m+1]
    y = s[n+m+1+1:n+m+1+n]
    v = s[n+m+1+n+1:n+m+1+n+m]
    w = s[n+m+1+n+m+1]
    stage_cost(x,y,u,v,h)
end

function Lx_func_alt(s)
    x = s[1:n]
    u = s[n+1:n+m]
    h = s[n+m+1]
    y = s[n+m+1+1:n+m+1+n]
    v = s[n+m+1+n+1:n+m+1+n+m]
    w = s[n+m+1+n+m+1]
    Lx_func(x,u,y,v,h)
end

function Lu_func_alt(s)
    x = s[1:n]
    u = s[n+1:n+m]
    h = s[n+m+1]
    y = s[n+m+1+1:n+m+1+n]
    v = s[n+m+1+n+1:n+m+1+n+m]
    w = s[n+m+1+n+m+1]
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    dxm = 2*h/8*fc_(x,u) - 2*h/8*fc_(y,v)
    L2h = 4/6*((h^2)*dxm'*Q*(xm - xf) + 2*h*el(xm,um))
    [Lu_func(x,u,y,v,h); 2*h/6*el(x,u) + L2h + 2*h/6*el(y,v)]
end

function Ly_func_alt(s)
    x = s[1:n]
    u = s[n+1:n+m]
    h = s[n+m+1]
    y = s[n+m+1+1:n+m+1+n]
    v = s[n+m+1+n+1:n+m+1+n+m]
    w = s[n+m+1+n+m+1]
    Ly_func(x,u,y,v,h)
end

function Lv_func_alt(s)
    x = s[1:n]
    u = s[n+1:n+m]
    h = s[n+m+1]
    y = s[n+m+1+1:n+m+1+n]
    v = s[n+m+1+n+1:n+m+1+n+m]
    w = s[n+m+1+n+m+1]
    [Lv_func(x,u,y,v,h);0]
end

function Lh_func_alt(s)
    x = s[1:n]
    u = s[n+1:n+m]
    h = s[n+m+1]
    y = s[n+m+1+1:n+m+1+n]
    v = s[n+m+1+n+1:n+m+1+n+m]
    w = s[n+m+1+n+m+1]
    xm = x_midpoint(x,y,fc_(x,u),fc_(y,v),h)
    um = u_midpoint(u,v)
    dxm = 2*h/8*fc_(x,u) - 2*h/8*fc_(y,v)
    L2h = 4/6*((h^2)*dxm'*Q*(xm - xf) + 2*h*el(xm,um))
    2/6*h*el(x,u) + L2h + 2/6*h*el(y,v)
end

s_alt = [x;u;h;y;v;0]
L_gradient_alt = ForwardDiff.gradient(stage_cost_alt,s_alt)

# Confirm that analytical minimum time expansions match ForwardDiff
@test isapprox(L_gradient_alt[1:n],_Lx)
@test isapprox(L_gradient_alt[n+1:n+m+1],_Lu)
@test isapprox(L_gradient_alt[n+m+1+1:n+m+1+n],_Ly)
@test isapprox(L_gradient_alt[n+m+1+n+1:n+m+1+n+m+1],_Lv)

@test isapprox(ForwardDiff.jacobian(Lx_func_alt,s_alt)[1:n,1:n],_Lxx)
@test isapprox(ForwardDiff.jacobian(Lu_func_alt,s_alt)[:,n+1:n+m+1],_Luu)
@test isapprox(ForwardDiff.jacobian(Ly_func_alt,s_alt)[:,n+m+1+1:n+m+1+n],_Lyy)
@test isapprox(ForwardDiff.jacobian(Lv_func_alt,s_alt)[:,n+m+1+n+1:n+m+1+n+m+1],_Lvv)

@test isapprox(ForwardDiff.jacobian(Lx_func_alt,s_alt)[1:n,n+1:n+m+1],_Lxu)
@test isapprox(ForwardDiff.jacobian(Lx_func_alt,s_alt)[1:n,n+m+1+1:n+m+1+n],_Lxy)
@test isapprox(ForwardDiff.jacobian(Lx_func_alt,s_alt)[1:n,n+m+1+n+1:n+m+1+n+m+1],_Lxv)
@test isapprox(ForwardDiff.jacobian(Lu_func_alt,s_alt)[1:m+1,n+m+1+1:n+m+1+n],_Luy)
@test isapprox(ForwardDiff.jacobian(Lu_func_alt,s_alt)[1:m+1,n+m+1+n+1:n+m+1+n+m+1],_Luv)
@test isapprox(ForwardDiff.jacobian(Ly_func_alt,s_alt)[1:n,n+m+1+n+1:n+m+1+n+m+1],_Lyv)
