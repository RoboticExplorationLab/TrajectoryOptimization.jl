# Setting up a Dynamics Model
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
```
# Overview
The Model type holds information about the dynamics of the system. All dynamics are assumed to be state-space models of the system of the form ẋ = f(x,u) where ẋ is the state derivative, x an n-dimentional state vector, and u in an m-dimensional control input vector. The function f can be any nonlinear function.

TrajectoryOptimization.jl poses the trajectory optimization problem by discretizing the state and control trajectories, which requires discretizing the dynamics, turning the continuous time differential equation into a discrete time difference equation of the form x[k+1] = f(x[k],u[k]), where k is the time step. There many methods of performing this discretization, and TrajectoryOptimization.jl offers several of the most common methods.

Sometimes is it convenient to write down the difference equation directly, rather than running a differential equation through a discretizing integration method. TrajectoryOptimization.jl offers method deal directly with either continuous differential equations, or discrete difference equations.

The `Model` type is parameterized by the `DynamicsType`, which is either `Continuous`, or `Discrete`. The models holds the equation f and it's Jacobian, ∇f, along with the dimensions of the state and control vectors.

Models can be created by writing down the dynamics analytically or be generated from a URDF file via [`RigidBodyDynamics.jl`](https://github.com/JuliaRobotics/RigidBodyDynamics.jl).

# Continuous Models
Continuous models assume differential equations are specified by an in-place function in one of the following forms:
```
f!(ẋ,x,u)
f!(ẋ,x,u,p)
```
and Jacobians of the form
```
∇f!(Z,x,u)
∇f!(Z,x,u,p)
```
where `ẋ` is the state derivative, `p` is a `NamedTuple` of model parameters, and `Z`is the (n × (n+m)) Jacobian matrix (i.e. [∇ₓf(x,u) ∇ᵤf(x,u)]). As soon as the model is created, however, only the forms without parameters (the top lines) are available. The `Model` type will automatically bake in the parameters. A new model must be created if a parameter is changed (this will be made easier in the future).


## Analytical Models
```@docs
AnalyticalModel
```
The following constructors can be used to create Continuous Analytical models
```@docs
Model(f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}=Dict{Symbol,Any}())
Model(f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}())
Model(f::Function, ∇f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}())
Model(f::Function, ∇f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}=Dict{Symbol,Any}())
```

## URDF Models
```@docs
RBDModel
```
The following constructors can be used to create a Model from a URDF file
```@docs
Model(mech::Mechanism, torques::Array)
Model(mech::Mechanism)
Model(urdf::String)
Model(urdf::String,torques::Array{Float64,1})
```

# Discrete Models
Discrete models assume difference equations are specified by an in-place function in one of the following forms:
```
f!(x′,x,u,dt)
f!(x′,x,u,p,dt)
```
and Jacobians of the form
```
∇f!(Z,x,u,dt)
∇f!(Z,x,u,p,dt)
```
where `x′` is the state at the next time step, `p` is a `NamedTuple` of model parameters, and `Z`is the (n × (n+m)) Jacobian matrix (i.e. [∇ₓf(x,u) ∇ᵤf(x,u)]).


## Analytical
An analytical model with discrete dynamics can be created using the following constructors
```@docs
AnalyticalModel{D}(f::Function, ∇f::Function, n::Int64, m::Int64,
          p::NamedTuple=NamedTuple(), d::Dict{Symbol,Any}=Dict{Symbol,Any}()
AnalyticalModel{D}(f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}=Dict{Symbol,Any}())
AnalyticalModel{D}(f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}())
```

## From Continuous Model
A discrete model can be created from a continuous model by specifying the integration (discretization) method. The following methods are currently supported

* midpoint
* rk3 (Third Order Runge-Kutta)
* rk4 (Fourth Order Runge-Kutta)

Use the following method to discretize a continuous model with one of the integration methods listed previously
```@docs
Model{Discrete}(model::Model{Continuous},discretizer::Function)
```

# API
```@docs
evaluate!(ẋ::AbstractVector,model::Model,x,u)
evaluate!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model,x,u)
jacobian!(Z::AbstractMatrix,ẋ::AbstractVector,model::Model,x,u)
jacobian!(Z::AbstractMatrix,model::Model,x,u)
evals(model::Model)
reset(model::Model)
```
