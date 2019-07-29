abstract type AbstractContinuousModel end
abstract type AbstractDiscreteModel <: AbstractModel end

struct SModel{D<:DynamicsType} <: AbstractModel
    f::Function
    ∇f::Function
    n::Int
    m::Int
    params::NamedTuple
    info::Dict{Symbol,Any}
end

# Continuous Model Constructors
function SModel{Continuous}(f::Function,n,m)
    ∇f = generate_jacobian_nip(f,n,m)
    params = NamedTuple()
    info = Dict{Symbol,Any}()
    SModel{Continuous}(f,∇f,n,m,params,info)
end

function SModel{Continuous}(f::Function,n,m,params)
    f_(x,u) = f(x,u,params)
    ∇f = generate_jacobian_nip(f_,n,m)
    params = NamedTuple()
    info = Dict{Symbol,Any}()
    SModel{Continuous}(f,∇f,n,m,params,info)
end

# Discrete Model Constructors
function SModel{Discrete}(f::Function,n,m)
    ∇f = generate_jacobian_nip(f,n,m,dt)
    params = NamedTuple()
    info = Dict{Symbol,Any}()
    SModel{Discrete}(f,∇f,n,m,params,info)
end

function SModel{Discrete}(f::Function,n,m,params)
    f_(x,u,dt) = f(x,u,dt,params)
    ∇f = generate_jacobian_nip(f_,n,m,dt)
    params = NamedTuple()
    info = Dict{Symbol,Any}()
    SModel{Discrete}(f,∇f,n,m,params,info)
end

# Dynamics Evaluations
dynamics(model::SModel{Continuous},x,u) = model.f(x,u)
dynamics(model::SModel{Discrete},x,u,dt) = model.f(x,u,dt)
jacobian(model::SModel{Continuous},x,u) = model.∇f(x,u)
jacobian(model::SModel{Discrete},x,u,dt) = model.∇f(x,u,dt)


function jacobian!(Z, model::SModel{Discrete},
        X::Vector{<:AbstractVector}, U::Vector{<:AbstractVector}, dt::Vector)
    for k in eachindex(U)
        Z[k] = jacobian(model,X[k],U[k],dt[k])
    end
end

function discretize(model::SModel{Continuous},integration::Symbol)
    if integration in [:rk3_nip]
        discretizer = eval(integration)
        fd = discretizer(model.f)
        ∇fd = generate_jacobian_nip(fd, model.n, model.m, 0.0)
        info = copy(model.info)
        info[:fc] = model.f
        info[:∇fc] = model.∇f
        SModel{Discrete}(fd, ∇fd, model.n, model.m, model.params, info)
    else
        error("Integration method not defined for $integration")
    end
end

function generate_jacobian_nip(f,n,m)
    ix,iu = 1:n, n .+ (1:m)
    f_aug(z) = f(view(z,ix), view(z,iu))
    ∇f(z) = ForwardDiff.jacobian(f_aug,z)
    ∇f(x::SVector,u::SVector) = ∇f([x;u])
    ∇f(x,u) = begin
        z = zeros(n+m)
        z[ix] = x
        z[iu] = u
        ∇f(z)
    end
    return ∇f
end

function generate_jacobian_nip(f,n,m,dt)
    ix = @SVector [i for i in 1:n]
    iu = @SVector [i for i in (n .+ (1:m))]
    idt = n+m+1
    z = zeros(n+m+1)
    @inbounds f_aug(z) = f(z[ix], z[iu], z[idt])
    ∇f(z) = ForwardDiff.jacobian(f_aug,z)
    ∇f(x::SVector,u::SVector,dt) = ∇f([x; u; @SVector [dt,]])
    @inbounds ∇f(x,u,dt) = begin
        z[ix] = x
        z[iu] = u
        z[idt] = dt
        ∇f(z)
    end
    return ∇f
end
