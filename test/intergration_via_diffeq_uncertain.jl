using ForwardDiff, OrdinaryDiffEq, BenchmarkTools

model = Dynamics.cartpole_model_uncertain
n = model.n; m = model.m; r = model.r
dt = 0.01

fd = midpoint_implicit_uncertain(model.f,n,m,r,dt)

x₊ = zeros(n)
x2₊ = zeros(n)

x = rand(n)
u = rand(m)

@benchmark fd(x₊,$x,$u,$zeros(r))

inds = (x=1:n,u=n .+ (1:m), dt=n+m+1, xx=(1:n,1:n),xu=(1:n,n .+ (1:m)), xdt=(1:n,n+m.+(1:1)))
s = zeros(n+m+r+1)
S0 = zeros(n,n+m+r+1)

fd_aug!(xdot,s) = fd(xdot,s[inds.x],s[inds.u],s[(n+m) .+ (1:r)],s[inds.dt])
Fd!(S,xdot,s) = ForwardDiff.jacobian!(S,fd_aug!,xdot,s)
∇fd!(x::AbstractVector,u::AbstractVector,w,dt::Float64) = begin
    s[inds.x] = x
    s[inds.u] = u
    s[(n+m) .+ (1:r)] = w
    s[inds.dt] = dt
    S0 = zeros(eltype(x),n,n+m+r+1)
    Fd!(S0,zero(x),s)
    return S0
end

F1 = ∇fd!(x,u,zeros(r),dt)

x3₊ = zeros(n)
model_d = discretize_model(model,:DiffEq_ImplicitMidpoint,dt)

model_d.f(x3₊,x,u,zeros(r))

isapprox(x3₊,x₊)

F3 = model_d.∇f(x,u,zeros(r),dt)

isapprox(F3,F1)
