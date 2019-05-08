import Base: copy, reset

abstract type DynamicsType end
abstract type Continuous <: DynamicsType end
abstract type Discrete <: DynamicsType end

abstract type ModelType end
abstract type Nominal <: ModelType end
abstract type Uncertain <: ModelType end

abstract type AbstractModel end
abstract type Model{M<:ModelType,D<:DynamicsType} <: AbstractModel end

"""
$(TYPEDEF)

Dynamics model

Holds all information required to uniquely describe a dynamic system, including
a general nonlinear dynamics function of the form `ẋ = f(x,u)`, where x ∈ ℜⁿ are
the states and u ∈ ℜᵐ are the controls.

Dynamics function, f, should be of the form
    f(ẋ,x,u,p) for Continuous models, where ẋ is the state derivative
    f(ẋ,x,u,p,dt) for Discrete models, where ẋ is the state at the next time step
    and x is the state vector, u is the control input vector, and p is an optional `NamedTuple` of static parameters (mass, gravity, etc.)

Dynamics jacobians, ∇f, should be of the form
    ∇f(Z,x,u,p) for Continuous models, and
    ∇f(Z,x,u,,p,dt) for discrete models
    where p is the same `NamedTuple` of parameters used in the dynamics
"""
struct AnalyticalModel{M,D} <: Model{M,D}
    f::Function   # dynamics f(ẋ,x,u)
    ∇f::Function  # dynamics jacobian
    n::Int        # number of states
    m::Int        # number of controls
    r::Int       # number of uncertain parameters
    params::NamedTuple
    evals::Vector{Int}
    info::Dict{Symbol,Any}

    """ $(SIGNATURES)
    Create a dynamics model given a dynamics function and Jacobian, with n states and m controls.

    Dynamics function should be of the form
        f(ẋ,x,u,p) for Continuous models, where ẋ is the state derivative
        f(ẋ,x,u,p,dt) for Discrete models, where ẋ is the state at the next time step
        and x is the state vector, u is the control input vector, and p is a `NamedTuple` of static parameters (mass, gravity, etc.)

    Optionally pass in a dictionary `d` with model information.
    `check_functions` option runs verification checks on the dynamics function and Jacobian to make sure they have the correct forms.
    """
    function AnalyticalModel{M,D}(f::Function, ∇f::Function, n::Int, m::Int, r::Int,
            p::NamedTuple=NamedTuple(), d::Dict{Symbol,Any}=Dict{Symbol,Any}();
            check_functions::Bool=false) where {M<:ModelType,D<:DynamicsType}
        d[:evals] = 0
        evals = [0,0]
        if check_functions && r == 0
            # Make dynamics inplace
            if is_inplace_dynamics(f,n,m,r)
                f! = f
            else
                f! = wrap_inplace(f)
            end
            #TODO new jacobian checks for uncertain model
            ∇f! = _check_jacobian(D,f,∇f,n,m)
            new{M,D}(f!,∇f!,n,m,r,p,evals,d)
        else
            new{M,D}(f,∇f,n,m,r,p,evals,d)
        end
    end
end

function AnalyticalModel{M,D}(f::Function, n::Int, m::Int, r::Int, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) where {M<:ModelType,D<:DynamicsType}
    p = NamedTuple()
    ∇f, = generate_jacobian(M,D,f,n,m)
    AnalyticalModel{M,D}(f,∇f,n,m,r,p,d)
end

function AnalyticalModel{Uncertain,D}(f::Function, n::Int, m::Int, r::Int, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) where {D<:DynamicsType}
    p = NamedTuple()
    ∇f, = generate_jacobian(Uncertain,D,f,n,m,r)
    AnalyticalModel{Uncertain,D}(f,∇f,n,m,r,p,d)
end

function AnalyticalModel{M,D}(f::Function, n::Int, m::Int, r::Int, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) where {M<:ModelType,D<:DynamicsType}
    f_p(ẋ,x,u) = f(ẋ,x,u,p)
    f_p(ẋ,x,u,p) = f(ẋ,x,u,p)
    ∇f, = generate_jacobian(M,D,f_p,n,m)
    AnalyticalModel{M,D}(f_p,∇f,n,m,r,p,d)
end

function AnalyticalModel{Uncertain,D}(f::Function, n::Int, m::Int, r::Int, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) where {D<:DynamicsType}
    f_p(ẋ,x,u,w) = f(ẋ,x,u,w,p)
    f_p(ẋ,x,u,w,p) = f(ẋ,x,u,w,p)
    ∇f, = generate_jacobian(Uncertain,D,f_p,n,m,r)
    AnalyticalModel{Uncertain,D}(f_p,∇f,n,m,r,p,d)
end


""" $(TYPEDSIGNATURES)
Create a dynamics model, using ForwardDiff to generate the dynamics jacobian, with parameters
Dynamics function passes in parameters:
    f(ẋ,x,u,p)
    where p in NamedTuple of parameters
"""
Model(f::Function, n::Int, m::Int, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) =
    AnalyticalModel{Nominal,Continuous}(f,n,m,0,d)

UncertainModel(f::Function, n::Int, m::Int, r::Int, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) =
    AnalyticalModel{Uncertain,Continuous}(f,n,m,r,d)

""" $(TYPEDSIGNATURES)
Create a dynamics model, using ForwardDiff to generate the dynamics jacobian, without parameters
Dynamics function of the form:
    f(ẋ,x,u)
"""
Model(f::Function, n::Int, m::Int, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) =
    AnalyticalModel{Nominal,Continuous}(f,n,m,0,p,d)

UncertainModel(f::Function, n::Int, m::Int, r::Int, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) =
    AnalyticalModel{Uncertain,Continuous}(f,n,m,p,r,d)

""" $(TYPEDSIGNATURES)
Create a dynamics model with an analytical Jacobian, with parameters
Dynamics functions pass in parameters:
    f(ẋ,x,u,p)
    ∇f(Z,x,u,p)
    where p in NamedTuple of parameters
"""
Model(f::Function, ∇f::Function, n::Int, m::Int, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) = begin
    f_p(ẋ,x,u) = f(ẋ,x,u,p)
    f_p(ẋ,x,u,p) = f(ẋ,x,u,p)
    ∇f_p(Z,x,u) = ∇f(Z,x,u,p)
    AnalyticalModel{Nominal,Continuous}(f_p,∇f_p,n,m,0,p,d, check_functions=true); end

UncertainModel(f::Function, ∇f::Function, n::Int, m::Int, r::Int, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) = begin
    f_p(ẋ,x,u,w) = f(ẋ,x,u,w,p)
    f_p(ẋ,x,u,w,p) = f(ẋ,x,u,w,p)
    ∇f_p(Z,x,u,w) = ∇f(Z,x,u,w,p)
    AnalyticalModel{Uncertain,Continuous}(f_p,∇f_p,n,m,r,p,d, check_functions=true); end

""" $(TYPEDSIGNATURES)
Create a dynamics model with an analytical Jacobian, without parameters
Dynamics functions pass of the form:
    f(ẋ,x,u)
    ∇f(Z,x,u)
"""
Model(f::Function, ∇f::Function, n::Int, m::Int, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) = begin
    p = NamedTuple()
    AnalyticalModel{Nominal,Continuous}(f,∇f,n,m,0,p,d, check_functions=true); end

UncertainModel(f::Function, ∇f::Function, n::Int, m::Int, r::Int, d::Dict{Symbol,Any}=Dict{Symbol,Any}()) = begin
    p = NamedTuple()
    AnalyticalModel{Uncertain,Continuous}(f,∇f,n,m,r,p,d, check_functions=true); end

""" $(SIGNATURES) Evaluate the dynamics at state `x` and control `x`
Keeps track of the number of evaluations
"""
function evaluate!(ẋ::AbstractVector,model::Model{M,Continuous},x::AbstractVector,u::AbstractVector) where M <: ModelType
    model.f(ẋ,x,u)
    model.evals[1] += 1
end

function evaluate!(ẋ::AbstractVector,model::Model{Uncertain,Continuous},x::AbstractVector,u::AbstractVector)
    model.f(ẋ,x,u,zeros(model.r))
    model.evals[1] += 1
end

function evaluate_uncertain!(ẋ::AbstractVector,model::Model{Uncertain,Continuous},x::AbstractVector,u::AbstractVector,w::AbstractVector)
    model.f(ẋ,x,u,w)
    model.evals[1] += 1
end

function evaluate!(ẋ::AbstractVector,model::Model{M,Discrete},x::AbstractVector,u::AbstractVector) where M <: ModelType
    model.f(ẋ,x,u)
    model.evals[1] += 1
end

function evaluate!(ẋ::AbstractVector,model::Model{Uncertain,Discrete},x::AbstractVector,u::AbstractVector)
    model.f(ẋ,x,u,zeros(model.r))
    model.evals[1] += 1
end

function evaluate_uncertain!(ẋ::AbstractVector,model::Model{Uncertain,Discrete},x::AbstractVector,u::AbstractVector,w::AbstractVector)
    model.f(ẋ,x,u,w)
    model.evals[1] += 1
end

function evaluate!(ẋ::AbstractVector,model::Model{M,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where {M <: ModelType,T}
    model.f(ẋ,x,u,dt)
    model.evals[1] += 1
end

function evaluate!(ẋ::AbstractVector,model::Model{Uncertain,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where T
    model.f(ẋ,x,u,zeros(model.r),dt)
    model.evals[1] += 1
end

function evaluate_uncertain!(ẋ::AbstractVector,model::Model{Uncertain,Discrete},x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T
    model.f(ẋ,x,u,w,dt)
    model.evals[1] += 1
end

""" $(SIGNATURES) Evaluate the dynamics and dynamics Jacobian simultaneously at state `x` and control `x`
Keeps track of the number of evaluations
"""
function evaluate!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{M,Continuous},x::AbstractVector,u::AbstractVector) where M <: ModelType
    model.∇f(Z,ẋ,x,u)
    model.evals[1] += 1
    model.evals[2] += 1
end

function evaluate!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{Uncertain,Continuous},x::AbstractVector,u::AbstractVector)
    idx = NamedTuple{(:x,:u)}((1:model.n,1:model.m))
    model.∇f(Z,ẋ[idx.x],x[idx.x],u[idx.u],zeros(model.r))
    model.evals[1] += 1
    model.evals[2] += 1
end

function evaluate_uncertain!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{Uncertain,Continuous},x::AbstractVector,u::AbstractVector,w::AbstractVector)
    idx = NamedTuple{(:x,:u)}((1:model.n,1:model.m))
    model.∇f(Z,ẋ[idx.x],x[idx.x],u[idx.u],w)
    model.evals[1] += 1
    model.evals[2] += 1
end

function evaluate!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{M,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where {M <: ModelType,T}
    model.∇f(Z,ẋ,x,u,dt)
    model.evals[1] += 1
    model.evals[2] += 1
end

function evaluate!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{Uncertain,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where {M <: ModelType,T}
    idx = NamedTuple{(:x,:u)}((1:model.n,1:model.m))
    model.∇f(Z,ẋ[idx.x],x[idx.x],u[idx.u],zeros(model.r),dt)
    model.evals[1] += 1
    model.evals[2] += 1
end

function evaluate_uncertain!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{Uncertain,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where {M <: ModelType,T}
    idx = NamedTuple{(:x,:u)}((1:model.n,1:model.m))
    model.∇f(Z,ẋ[idx.x],x[idx.x],u[idx.u],w,dt)
    model.evals[1] += 1
    model.evals[2] += 1
end

""" $(SIGNATURES) Evaluate the dynamics and dynamics Jacobian simultaneously at state `x` and control `x`
Keeps track of the number of evaluations
"""
jacobian!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{M,Continuous},x::AbstractVector,u::AbstractVector) where M <: ModelType = evaluate!(Z,ẋ,model,x,u)
jacobian!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{Uncertain,Continuous},x::AbstractVector,u::AbstractVector) = evaluate!(Z,ẋ,model,x,u,zeros(model.r))
jacobian_uncertain!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{Uncertain,Continuous},x::AbstractVector,u::AbstractVector,w::AbstractVector) = evaluate!(Z,ẋ,model,x,u,w)

jacobian!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{M,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where {M <: ModelType,T} = evaluate!(Z,ẋ,model,x,u,dt)
jacobian!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{Uncertain,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where T = evaluate!(Z,ẋ,model,x,u,zeros(model.r),dt)
jacobian_uncertain!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model{Uncertain,Discrete},x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T = evaluate!(Z,ẋ,model,x,u,w,dt)


""" $(SIGNATURES) Evaluate the dynamics Jacobian simultaneously at state `x` and control `x`
Keeps track of the number of evaluations
"""
function jacobian!(Z::AbstractMatrix,model::Model{M,Continuous},x::AbstractVector,u::AbstractVector) where M <: ModelType
    model.∇f(Z,x,u)
    model.evals[2] += 1
end

function jacobian!(Z::AbstractMatrix,model::Model{Uncertain,Continuous},x::AbstractVector,u::AbstractVector)
    model.∇f(Z,x,u,zeros(model.r))
    model.evals[2] += 1
end

function jacobian_uncertain!(Z::AbstractMatrix,model::Model{Uncertain,Continuous},x::AbstractVector,u::AbstractVector,w::AbstractVector)
    model.∇f(Z,x,u,w)
    model.evals[2] += 1
end

function jacobian!(Z::AbstractMatrix,model::Model{M,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where {M <: ModelType,T}
    model.∇f(Z,x,u,dt)
    model.evals[2] += 1
end

function jacobian!(Z::AbstractMatrix,model::Model{Uncertain,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where T
    model.∇f(Z,x,u,zeros(model.r),dt)
    model.evals[2] += 1
end

function jacobian_uncertain!(Z::AbstractMatrix,model::Model{Uncertain,Discrete},x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T
    model.∇f(Z,x,u,w,dt)
    model.evals[2] += 1
end

function jacobian!(Z::PartedMatTrajectory{T},model::Model{M,Discrete},X::VectorTrajectory{T},U::VectorTrajectory{T},dt::T) where {M<:ModelType,T}
    N = length(X)
    for k = 1:N-1
        jacobian!(Z[k],model,X[k],U[k],dt)
    end
end

function jacobian!(Z::PartedMatTrajectory{T},model::Model{Uncertain,Discrete},X::VectorTrajectory{T},U::VectorTrajectory{T},dt::T) where T
    N = length(X)
    for k = 1:N-1
        jacobian!(Z[k],model,X[k],U[k],dt)
    end
end

function jacobian_uncertain!(Z::PartedMatTrajectory{T},model::Model{Uncertain,Discrete},X::VectorTrajectory{T},U::VectorTrajectory{T},W::VectorTrajectory{T},dt::T) where T
    N = length(X)
    for k = 1:N-1
        jacobian!(Z[k],model,X[k],U[k],W[k],dt)
    end
end



""" $(SIGNATURES) Return the number of dynamics evaluations """
evals(model::Model) = model.evals[1]

""" $(SIGNATURES) Reset the evaluation counts for the model """
reset(model::Model) = begin model.evals[1] = 0; return nothing end

Base.length(model::Model{M,Discrete}) where M<:ModelType = model.n + model.m + 1
Base.length(model::Model{M,Continuous}) where M<:ModelType = model.n + model.m

Base.length(model::Model{Uncertain,Discrete}) = model.n + model.m + model.r + 1
Base.length(model::Model{Uncertain,Continuous}) = model.n + model.m + model.r

PartedArrays.create_partition(model::Model{M,Discrete}) where M <:ModelType = create_partition((model.n,model.m,1),(:x,:u,:dt))
PartedArrays.create_partition2(model::Model{M,Discrete}) where M <:ModelType = create_partition2((model.n,),(model.n,model.m,1),(:x,),(:x,:u,:dt))
PartedArrays.create_partition(model::Model{M,Continuous}) where M <:ModelType = create_partition((model.n,model.m),(:x,:u))
PartedArrays.create_partition2(model::Model{M,Continuous}) where M <:ModelType = create_partition2((model.n,),(model.n,model.m),(:x,),(:x,:u))

PartedArrays.create_partition(model::Model{Uncertain,Discrete}) = create_partition((model.n,model.m,model.r,1),(:x,:u,:w,:dt))
PartedArrays.create_partition2(model::Model{Uncertain,Discrete}) = create_partition2((model.n,),(model.n,model.m,model.r,1),(:x,),(:x,:u,:w,:dt))
PartedArrays.create_partition(model::Model{Uncertain,Continuous}) = create_partition((model.n,model.m,model.r),(:x,:u,:w))
PartedArrays.create_partition2(model::Model{Uncertain,Continuous}) = create_partition2((model.n,),(model.n,model.m,model.r),(:x,),(:x,:u,:w))

PartedArrays.BlockVector(model::Model) = BlockArray(zeros(length(model)),create_partition(model))
PartedArrays.BlockVector(T::Type,model::Model) = BlockArray(zeros(T,length(model)),create_partition(model))
PartedArrays.BlockMatrix(model::Model) = BlockArray(zeros(model.n,length(model)),create_partition2(model))
PartedArrays.BlockMatrix(T::Type,model::Model) = BlockArray(zeros(T,model.n,length(model)),create_partition2(model))


function dynamics(model::Model{M,D},xdot::AbstractVector,x::AbstractVector,u::AbstractVector) where {M<:ModelType,D<:DynamicsType}
    model.f(xdot,x,u)
    model.evals[1] += 1
end

#TODO
function dynamics(model::Model{Uncertain,D},xdot::AbstractVector,x::AbstractVector,u::AbstractVector) where D <: DynamicsType
    model.f(view(xdot.x),x.x,u.u,0.)
    model.evals[1] += 1
end

"""
$(TYPEDEF)
RigidBodyDynamics model. Wrapper for a RigidBodyDynamics Mechanism
"""
struct RBDModel{M,D} <: Model{M,D}
    f::Function # continuous dynamics (ie, differential equation)
    n::Int # number of states
    m::Int # number of controls
    r::Int # number of uncertain parameters
    mech::Mechanism  # RigidBodyDynamics Mechanism
    evals::Vector{Int}
    info::Dict{Symbol,Any}
end



"""
$(SIGNATURES)
Model(mech::Mechanism, torques::Array{Bool, 1}) Constructor for an underactuated mechanism, where torques is a binary array
that specifies whether a joint is actuated.
"""
function Model(mech::Mechanism, torques::Array)

    # construct a model using robot dynamics equation assembed from URDF file
    n = num_positions(mech) + num_velocities(mech) + num_additional_states(mech)
    num_joints = length(joints(mech))  # subtract off joint to world

    if length(torques) != num_joints
        error("Torque underactuation specified does not match mechanism dimensions")
    end

    m = convert(Int,sum(torques)) # number of actuated (ie, controllable) joints
    torque_matrix = 1.0*Matrix(I,num_joints,num_joints)[:,torques.== 1] # matrix to convert from control inputs to mechanism joints

    statecache = StateCache(mech)
    dynamicsresultscache = DynamicsResultCache(mech)

    function f(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T}) where T
        state = statecache[T]
        dyn = dynamicsresultscache[T]
        dynamics!(view(ẋ,1:n), dyn, state, x, torque_matrix*u)
        return nothing
    end
    d = Dict{Symbol,Any}()

    evals = [0,]
    RBDModel{Nominal,Continuous}(f, n, m, 0, mech, evals, d)
end

"""
$(SIGNATURES)
 Construct model from a `Mechanism` type from `RigidBodyDynamics`
 """
function Model(mech::Mechanism)
    m = length(joints(mech))  # subtract off joint to world
    Model(mech,ones(m))
end


"""$(SIGNATURES) Construct a fully actuated model from a string to a urdf file"""
function Model(urdf::String)
    # construct model using string to urdf file
    mech = parse_urdf(urdf)
    Model(mech)
end

"""$(SIGNATURES) Construct a partially actuated model from a string to a urdf file, where torques is a binary array that specifies whether a joint is actuated."""
function Model(urdf::String,torques::Array{Float64,1})
    # underactuated system (potentially)
    mech = parse_urdf(urdf)
    Model(mech,torques)
end

#TODO uncertain version for continous model
generate_jacobian(f!::Function,n::Int,m::Int,p::Int=n) = generate_jacobian(Nominal,Continuous,f!,n,m,p)

function generate_jacobian(::Type{Nominal},::Type{Continuous},f!::Function,n::Int,m::Int,p::Int=n)
    inds = (x=1:n,u=n .+ (1:m), px=(1:p,1:n),pu=(1:p,n .+ (1:m)))
    Z = BlockArray(zeros(p,n+m),inds)
    z = zeros(n+m)
    v0 = zeros(p)
    f_aug(dZ::AbstractVector,z::AbstractVector) = f!(dZ,view(z,inds.x), view(z,inds.u))
    ∇fz(Z::AbstractMatrix,v::AbstractVector,z::AbstractVector) = ForwardDiff.jacobian!(Z,f_aug,v,z)
    ∇f!(Z::AbstractMatrix,v::AbstractVector,x::AbstractVector,u::AbstractVector) = begin
        z[inds.x] = x
        z[inds.u] = u
        ∇fz(Z,v,z)
        return nothing
    end
    ∇f!(Z::AbstractMatrix,x::AbstractVector,u::AbstractVector) = begin
        z[inds.x] = x
        z[inds.u] = u
        ∇fz(Z,v0,z)
        return nothing
    end
    ∇f!(x::AbstractVector,u::AbstractVector) = begin
        z[inds.x] = x
        z[inds.u] = u
        ∇fz(Z,v0,z)
        return Z
    end
    return ∇f!, f_aug
end

function generate_jacobian(::Type{Nominal},::Type{Discrete},fd!::Function,n::Int,m::Int)
    inds = (x=1:n,u=n .+ (1:m), dt=n+m+1, xx=(1:n,1:n),xu=(1:n,n .+ (1:m)), xdt=(1:n,n+m.+(1:1)))
    S0 = zeros(n,n+m+1)
    s = zeros(n+m+1)
    ẋ0 = zeros(n)

    fd_aug!(xdot,s) = fd!(xdot,view(s,inds.x),view(s,inds.u),s[inds.dt])
    Fd!(S,xdot,s) = ForwardDiff.jacobian!(S,fd_aug!,xdot,s)
    ∇fd!(S::AbstractMatrix,ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,dt::T) where T = begin
        s[inds.x] = x
        s[inds.u] = u
        s[inds.dt] = dt
        Fd!(S,ẋ,s)
        return nothing
    end
    ∇fd!(S::AbstractMatrix,x::AbstractVector,u::AbstractVector,dt::T) where T = begin
        s[inds.x] = x
        s[inds.u] = u
        s[inds.dt] = dt
        Fd!(S,ẋ0,s)
        return nothing
    end
    ∇fd!(x::AbstractVector,u::AbstractVector,dt::T) where T = begin
        s[inds.x] = x
        s[inds.u] = u
        s[inds.dt] = dt
        Fd!(S0,ẋ0,s)
        return S0
    end
    return ∇fd!, fd_aug!
end

function generate_jacobian(::Type{Uncertain},::Type{Continuous},f!::Function,n::Int,m::Int,r::Int)
    inds = (x=1:n, u=n .+ (1:m),w = (n+m) .+ (1:r), xx=(1:n,1:n),xu=(1:n,n .+ (1:m)),xw=(1:n,(n+m) .+ (1:r)))
    S0 = zeros(n,n+m+r)
    s = zeros(n+m+r)
    ẋ0 = zeros(n)

    f_aug!(xdot,s) = f!(xdot,view(s,inds.x),view(s,inds.u),view(s,inds.w))
    F!(S,xdot,s) = ForwardDiff.jacobian!(S,f_aug!,xdot,s)
    ∇f!(S::AbstractMatrix,ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,w::AbstractVector) = begin
        s[inds.x] = x
        s[inds.u] = u
        s[inds.w] = w
        F!(S,ẋ,s)
        return nothing
    end
    ∇f!(S::AbstractMatrix,x::AbstractVector,u::AbstractVector,w::AbstractVector) = begin
        s[inds.x] = x
        s[inds.u] = u
        s[inds.w] = w
        F!(S,ẋ0,s)
        return nothing
    end
    ∇f!(x::AbstractVector,u::AbstractVector,w::AbstractVector) = begin
        s[inds.x] = x
        s[inds.u] = u
        s[inds.w] = w
        Fd!(S0,ẋ0,s)
        return S0
    end
    return ∇f!, f_aug!
end

function generate_jacobian(::Type{Uncertain},::Type{Discrete},fd!::Function,n::Int,m::Int,r::Int)
    inds = (x=1:n, u=n .+ (1:m),w = (n+m) .+ (1:r), dt=n+m+r+1, xx=(1:n,1:n),xu=(1:n,n .+ (1:m)),xw=(1:n,(n+m) .+ (1:r)), xdt=(1:n,(n+m+r) .+ (1:1)))
    S0 = zeros(n,n+m+r+1)
    s = zeros(n+m+r+1)
    ẋ0 = zeros(n)

    fd_aug!(xdot,s) = fd!(xdot,view(s,inds.x),view(s,inds.u),view(s,inds.w),s[inds.dt])
    Fd!(S,xdot,s) = ForwardDiff.jacobian!(S,fd_aug!,xdot,s)
    ∇fd!(S::AbstractMatrix,ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T= begin
        s[inds.x] = x
        s[inds.u] = u
        s[inds.w] = w
        s[inds.dt] = dt
        Fd!(S,ẋ,s)
        return nothing
    end
    ∇fd!(S::AbstractMatrix,x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T = begin
        s[inds.x] = x
        s[inds.u] = u
        s[inds.w] = w
        s[inds.dt] = dt
        Fd!(S,ẋ0,s)
        return nothing
    end
    ∇fd!(x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T = begin
        s[inds.x] = x
        s[inds.u] = u
        s[inds.w] = w
        s[inds.dt] = dt
        Fd!(S0,ẋ0,s)
        return S0
    end
    return ∇fd!, fd_aug!
end

"""$(SIGNATURES)
Convert a continuous dynamics model into a discrete one using the given discretization function.
    The discretization function can either be one of the currently supported functions (midpoint, rk3, rk4) or a custom method that has the following form
    ```
    function discretizer(f::Function,dt::Float64)
        function fd!(xdot,x,u,dt)
            # Your code
            return nothing
        end
        return fd!
    end
    ```
"""

function discretize_model(model::Model{M,Continuous},discretizer::Symbol, dt::T=1.0) where {M<:ModelType,T}
    fd!,∇fd! = discretize(model.f,discretizer,dt,model.n,model.m)
    info_d = deepcopy(model.info)
    integration = string(discretizer)
    info_d[:integration] = Symbol(replace(integration,"TrajectoryOptimization." => ""))
    info_d[:fc] = model.f
    info_d[:∇fc] = model.∇f
    AnalyticalModel{M,Discrete}(fd!, ∇fd!, model.n, model.m, model.r, model.params, info_d)
end

function discretize_model(model::Model{Uncertain,Continuous},discretizer::Symbol,dt::T=1.0) where T
    fd!,∇fd! = discretize_uncertain(model.f,discretizer,dt,model.n,model.m,model.r)
    info_d = deepcopy(model.info)
    integration = string(discretizer)
    info_d[:integration] = Symbol(replace(integration,"TrajectoryOptimization." => ""))
    info_d[:fc] = model.f
    info_d[:∇fc] = model.∇f
    AnalyticalModel{Uncertain,Discrete}(fd!, ∇fd!, model.n, model.m, model.r, model.params, info_d)
end

midpoint(model::Model{M,Continuous},dt::T) where {M <: ModelType,T} = discretize_model(model, :midpoint, dt)
rk3(model::Model{M,Continuous},dt::T) where {M <: ModelType,T} = discretize_model(model, :rk3, dt)
rk4(model::Model{M,Continuous},dt::T) where {M <:ModelType,T} = discretize_model(model, :rk4, dt)

function discretize(f::Function,discretization::Symbol,dt::T,n::Int,m::Int) where T
    discretizer = eval(discretization)
    fd! = discretizer(f,dt)
    ∇fd!, = generate_jacobian(Nominal,Discrete,fd!,n,m)
    return fd!,∇fd!
end

function discretize_uncertain(f::Function,discretization::Symbol,dt::T,n::Int,m::Int,r::Int) where T
    discretizer = eval(Symbol(String(discretization) * "_uncertain"))
    fd! = discretizer(f,dt)
    ∇fd!, = generate_jacobian(Uncertain,Discrete,fd!,n,m,r)
    return fd!,∇fd!
end




"""
$(SIGNATURES)
Determine if the dynamics in model are in place. i.e. the function call is of
the form `f!(xdot,x,u)`, where `xdot` is modified in place. Returns a boolean.
"""
function is_inplace_dynamics(model::Model)::Bool
    is_inplace_dynamics(model.f,model.n,model.m,model.r)
    # x = rand(model.n)
    # u = rand(model.m)
    # xdot = rand(model.n)
    # if model.r == 0
    #     try
    #         model.f(xdot,x,u)
    #     catch x
    #         if x isa MethodError
    #             return false
    #         end
    #     end
    # else
    #     w = rand(model.r)
    #     try
    #         model.f(xdot,x,u,w)
    #     catch x
    #         if x isa MethodError
    #             return false
    #         end
    #     end
    # end
    #
    # return true
end

function is_inplace_dynamics(f::Function,n::Int,m::Int,r::Int)::Bool
    x = rand(n)
    u = rand(m)
    xdot = rand(n)
    if r == 0
        try
            f(xdot,x,u)
        catch x
            if x isa MethodError
                return false
            end
        end
    else
        w = rand(r)
        try
            f(xdot,x,u,w)
        catch x
            if x isa MethodError
                return false
            end
        end
    end
    return true
end

"""
$(SIGNATURES)
Makes the dynamics function `f(x,u)` appear to operate as an inplace operation of the
form `f!(xdot,x,u)`.
"""
function wrap_inplace(f::Function)
    f!(xdot,x,u) = copyto!(xdot, f(x,u))
    f!(xdot,x) = copyto!(xdot, f(x))
end

"""$(SIGNATURES)
Checks jacobians of functions of the form `f(v,x,u)` to make sure they have the correct forms.
Jacobians should have the following three forms
```∇f(x,u)
∇f(Z,x,u)
∇f(Z,v,x,u)```
"""
function _test_jacobian(::Type{Continuous},∇f::Function)
    form = [true,true,true]
    form[1] = hasmethod(∇f,(AbstractVector,AbstractVector))
    form[2] = hasmethod(∇f,(AbstractMatrix,AbstractVector,AbstractVector))
    form[3] = hasmethod(∇f,(AbstractMatrix,AbstractVector,AbstractVector,AbstractVector))
    return form
end

"""$(SIGNATURES)
Checks jacobians of functions of the form `f(v,x,u,dt)` to make sure they have the correct forms.
Jacobians should have the following three forms
```∇f(x,u,dt)
∇f(S,x,u,dt)
∇f(S,v,x,u,dt)```
"""
function _test_jacobian(::Type{Discrete},∇f::Function)
    form = [true,true,true]
    form[1] = hasmethod(∇f,(AbstractVector,AbstractVector,Float64))
    form[2] = hasmethod(∇f,(AbstractMatrix,AbstractVector,AbstractVector,Float64))
    form[3] = hasmethod(∇f,(AbstractMatrix,AbstractVector,AbstractVector,AbstractVector,Float64))
    return form
end

_check_jacobian(f::Function,∇f::Function,n::Int,m::Int,p::Int=n) = _check_jacobian(Continuous,f,∇f,n,m,p)
function _check_jacobian(::Type{Continuous},f::Function,∇f::Function,n::Int,m::Int,p::Int=n)
    forms = _test_jacobian(Continuous,∇f)
    if !forms[2]
        throw("Jacobians must have the method ∇f(Z,x,u)")
    else
        inds = (x=1:n,u=n .+ (1:m), px=(1:p,1:n),pu=(1:p,n .+ (1:m)))
        Z = BlockArray(zeros(p,n+m),inds)
        z = zeros(n+m)

        # Copy the correct method form
        ∇f!(Z,x,u) = ∇f(Z,x,u)

        # Implement the missing method(s)
        if forms[1]
            ∇f!(x,u) = ∇f(x,u)
        else
            ∇f!(x,u) = begin
                ∇f(Z,x,u)
                return Z
            end
        end
        if forms[3]
            ∇f!(Z,v,x,u) = ∇f(Z,v,x,u)
        else
            ∇f!(Z,v,x,u) = begin
                x = z[inds.x]
                u = z[inds.u]
                f(v,x,u)
                ∇f(Z,x,u)
            end
        end
    end
    return ∇f!

end

function _check_jacobian(::Type{Discrete},f::Function,∇f::Function,n::Int,m::Int,p::Int=n)
    forms = _test_jacobian(Discrete,∇f)
    if !forms[2]
        throw("Jacobians must have the method ∇f(Z,x,u,dt)")
    else
        inds = (x=1:n,u=n .+ (1:m), dt=n+m+1, xx=(1:n,1:n),xu=(1:n,n .+ (1:m)), xdt=(1:n,n+m.+(1:1)))
        S = BlockArray(zeros(p,n+m+1),inds)
        s = zeros(n+m+1)

        # Copy the correct method form
        ∇f!(S,x,u,dt) = ∇f(S,x,u,dt)
         # ∇f! = ∇f


        # Implement the missing method(s)
        if forms[1]
            ∇f!(x,u,dt) = ∇f(x,u,dt)
        else
            ∇f!(x,u,dt) = begin
                ∇f!(S,x,u,dt)
                return S
            end
        end
        if forms[3]
            ∇f!(S,v,x,u,dt) = ∇f(S,v,x,u,dt)
        else
            ∇f!(S,v,x,u,dt) = begin
                x = s[inds.x]
                u = s[inds.u]
                dt = s[inds.dt]
                f(v,x,u,dt)
                ∇f!(S,x,u,dt)
            end
        end
    end
    return ∇f!
end

function _check_dynamics(f::Function,n::Int,m::Int)
    no_params = hasmethod(f,(AbstractVector,AbstractVector,AbstractVector))
    with_params = hasmethod(f,(AbstractVector,AbstractVector,AbstractVector,Any))
    return [no_params,with_params]
end
