using ForwardDiff, OrdinaryDiffEq, BenchmarkTools

model = Dynamics.cartpole_model
n = model.n; m = model.m
dt = 0.01

fd = midpoint_implicit(model.f,n,m,dt)

x₊ = zeros(n)
x2₊ = zeros(n)

x = rand(n)
u = rand(m)

@benchmark fd(x₊,$x,$u)

function f_aug(z,p,t)
    ż = zero(z)
    model.f(view(ż,1:n),z[1:n],z[n .+ (1:m)])
    ż
end

b = :DiffEq_ImplicitMidpoint
c = String(b)
c[1:6] == "DiffEq"
d = split(c,"_")[2]
integrator = eval(Symbol(d))()

typeof(integrator)

function fd_ode(y,x,u,dt=dt)
    _tf = dt
    _t0 = 0.

    u0=vec([x;u])
    tspan = (_t0,_tf)
    pro = ODEProblem(f_aug,u0,tspan)
    sol = OrdinaryDiffEq.solve(pro,integrator,dt=dt)
    copyto!(y,sol.u[end][1:n])
end

@benchmark fd_ode(x2₊,$x,$u)

isapprox(x₊,x2₊)

inds = (x=1:n,u=n .+ (1:m), dt=n+m+1, xx=(1:n,1:n),xu=(1:n,n .+ (1:m)), xdt=(1:n,n+m.+(1:1)))
S0 = zeros(n,n+m+1)

fd_aug!(xdot,s) = fd(xdot,s[inds.x],s[inds.u],s[inds.dt])
Fd!(S,xdot,s) = ForwardDiff.jacobian!(S,fd_aug!,xdot,s)
∇fd!(x::AbstractVector,u::AbstractVector,dt::Float64) = begin
    s[inds.x] = x
    s[inds.u] = u
    s[inds.dt] = dt
    S0 = zeros(eltype(x),n,n+m+1)
    Fd!(S0,zero(x),s)
    return S0
end

F1 = ∇fd!(x,u,dt)

_S0 = zeros(n,n+m+1)

fd_ode_aug!(xdot,s) = fd_ode(xdot,s[inds.x],s[inds.u],s[inds.dt])
Fd_ode!(S,xdot,s) = ForwardDiff.jacobian!(S,fd_ode_aug!,xdot,s)
∇fd_ode!(x::AbstractVector,u::AbstractVector,dt::Float64) = begin
    s[inds.x] = x
    s[inds.u] = u
    s[inds.dt] = dt
    _S0 = zeros(eltype(x),n,n+m+1)
    Fd_ode!(_S0,zero(x),s)
    return _S0
end

F2 = ∇fd_ode!(x,u,dt)

isapprox(F1,F2)


#
x3₊ = zeros(n)
model_d = discretize_model(model,:DiffEq_ImplicitMidpoint,dt)

model_d.f(x3₊,x,u)

isapprox(x3₊,x2₊)

F3 = model_d.∇f(x,u,dt)

isapprox(F3,F2)
