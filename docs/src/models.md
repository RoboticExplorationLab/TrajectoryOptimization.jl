# [1. Setting up a Dynamics Model](@id model_section)
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["models.md"]
```
## Overview
The Model type holds information about the dynamics of the system. All dynamics are assumed to be state-space models of the system of the form y = f(x,u) where y is the state derivative (Continous) or the next state (Discrete), x an n-dimentional state vector, and u in an m-dimensional control input vector. The function f can be any nonlinear function.

TrajectoryOptimization.jl $\textit{discrete}$ trajectory optimization problem by discretizing the state and control trajectories, which requires discretizing the dynamics, i.e., turning the continuous time differential equation into a discrete time difference equation of the form x[k+1] = f(x[k],u[k]), where k is the time step. There many methods of performing this discretization, and TrajectoryOptimization.jl offers several of the most common methods.

Sometimes is it convenient to write down the difference equation directly, rather than running a differential equation through a discretizing integration method. TrajectoryOptimization.jl offers method deal directly with either continuous differential equations, or discrete difference equations.

The `Model` type is parameterized by the `DynamicsType`, which is either `Continuous`, or `Discrete`. The models holds the equation f and its Jacobian, ∇f, along with the dimensions of the state and control vectors.

Models can be created by writing down the dynamics analytically or be generated from a URDF file via [`RigidBodyDynamics.jl`](https://github.com/JuliaRobotics/RigidBodyDynamics.jl).

## Continuous Models

### From analytical function
Let's start by writing down a dynamics function for a simple pendulum with state [θ; ω] and a torque control input
```julia
function pendulum_dynamics!(ẋ,x,u)
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    J = 0.25
    g = 9.81
    ẋ[1] = x[2]
    ẋ[2] = (u[1] - m*g*lc*sin(x[1]) - b*x[2])/J
end
```
Note that the function is in-place, in that it writes the result to the first argument. It is also good practice to concretely specify the location to write to rather than using something like `ẋ[1:end]` or `ẋ[:]`.

Notice that we had to specify a handful of constants when writing down the dynamics. We could have initialized them outside the scope of the function (which may result in global variables, so be careful!) or we can pass them in as a `NamedTuple` of parameters:
```julia
function pendulum_dynamics_params!(xdot,x,u,p)
    xdot[1] = x[2]
    xdot[2] = (u[1] - p.m * p.g * p.lc * sin(x[1]) - p.b*x[2])/p.J
end
```
We can now create our model using our analytical dynamics function with or without the parameters tuple
```julia
n,m = 2,1
model = Model(pendulum_dynamics!, n, m)

params = (m=1, l=0.5, b=0.1, lc=0.5, J=0.25, g=9.81)
model = Model(pendulum_dynamics_params!, n, m, params)
```

### With analytical Jacobians
Since we have a very simple model, writing down an analytical expression of the Jacobian is straightforward:
```julia
function pendulum_jacobian!(Z,x,u)
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    J = 0.25
    g = 9.81

    Z[1,1] = 0                    # ∂θdot/∂θ
    Z[1,2] = 1                    # ∂θdot/∂ω
    Z[1,3] = 0                    # ∂θ/∂u
    Z[2,1] = -m*g*lc*cos(x[1])/J  # ∂ωdot/∂θ
    Z[2,2] = -b/J                 # ∂ωdot/∂ω
    Z[2,3] = 1/J                  # ∂ωdot/∂u
end

function pendulum_jacobian_params!(Z,x,u,p)
    Z[1,1] = 0                                    # ∂θdot/∂θ
    Z[1,2] = 1                                    # ∂θdot/∂ω
    Z[1,3] = 0                                    # ∂θ/∂u
    Z[2,1] = -p.m * p.g * p.lc * cos(x[1]) / p.J  # ∂ωdot/∂θ
    Z[2,2] = -p.b / p.J                           # ∂ωdot/∂ω
    Z[2,3] = 1/p.J                                # ∂ωdot/∂u
end
```
We can then pass these functions into the model instead of using `ForwardDiff` to calculate them
```julia
model = Model(pendulum_dynamics!, pendulum_jacobian!, n, m)
model = Model(pendulum_dynamics_params!, pendulum_jacobian_params!, n, m, params)
```


### URDF Files
Instead of writing down the dynamics explicity, we can import the dynamics from geometry specified in a URDF model using `RigidBodyDynamics.jl`. Let's say we have a URDF file for a double pendulum and don't want to bother writing down the dynamics, then we can create a model using any of the following methods
```julia
using RigidBodyDynamics
# From a string
urdf = "doublependulum.urdf"
model = Model(urdf)

# From a RigidBodyDynamics `Mechanism` type
mech = parse_urdf(urdf)  # return a Mechanism type
model = Model(mech)
```
Now let's say we want to control an underactuated `acrobot`, which can only control the second joint. We can pass in a vector of Booleans to specify which of the joints are "active."

```julia
joints = [false,true]

# From a string
urdf = "doublependulum.urdf"
model = Model(urdf,joints)

# From a RigidBodyDynamics `Mechanism` type
mech = parse_urdf(urdf)  # return a Mechanism type
model = Model(mech,joints)
```

### A note on Model types
While the constructors look very similar, URDF models actually return a slightly different type than the the analytical ones of the first section. Analytical models are represented by
```@docs
AnalyticalModel
```
whereas those created from a URDF are represented by
```@docs
RBDModel
```
which explicitly stores the `Mechanism` internally.



## Discrete Models
The previous methods all generate models with continuous dynamics (note that all of the models returned above will be of the type `Model{Continuous}`. In order to perform trajectory optimization we need to have discrete dynamics. Typically, we will form the continuous dynamics as we did above and then use a particular integration scheme to discretize it. Alternatively, we may know the analytical expression for the discrete dynamics.

### From a continuous model
Assuming we have a model of type `Model{Continuous}`, we can discretize as follows:
```julia
model_discrete = Model{Discrete}(model,discretizer)
```
where `discretizer` is a function that returns a discretized version of the continuous dynamics. TrajectoryOptimization.jl offers the following integration schemes

* midpoint
* rk3 (Third Order Runge-Kutta)
* rk4 (Fourth Order Runge-Kutta)

To create a discrete model of the pendulum with fourth order Runge-Kutta integration we would do the following
```julia
# Create the continuous model (any of the previously mentioned methods would work here)
params = (m=1, l=0.5, b=0.1, lc=0.5, J=0.25, g=9.81)
model = Model(pendulum_dynamics_params!, n, m, params)

# Discretize the continuous model
model_discrete = Model{Discrete}(model,rk4)
```

### From an analytical expression
Very little changes when specifying an analytical discrete model. The only change is that both the dynamics and Jacobian functions must take in the time step `dt` as an argument. Here is an example for the pendulum using Euler integration for simplicity
```julia
function pendulum_discrete!(xdot,x,u,dt)
    pendulum_dynamics!(xdot,x,u)
    xdot .= x + xdot*dt
end
```
The Jacobian is similarly specified as a function of the form `∇f(Z,x,u,dt)`. We don't give an example for brevity.

The model is then created in a similar fashion to the above methods
```julia
model_discrete = AnalyticalModel{Discrete}(pendulum_discrete!,n,m)
model_discrete = AnalyticalModel{Discrete}(pendulum_discrete_params!,n,m,params)  # if we defined this

# If we defined the Jacobian function we also create it as
model_discrete = AnalyticalModel{Discrete}(pendulum_discrete!, pendulum_discrete_jacobian!, n, m)
model_discrete = AnalyticalModel{Discrete}(pendulum_discrete_params!, pendulum_discrete_jacobian!, n, m, params)
```


## Methods
Models are pretty basic types and don't offer much functionality other than specifying the dynamics. We can get the number of stage and controls as follows
```julia
n = model.n
m = model.m
```

### Testing the dynamics
It's often useful to test out the dynamics or its Jacobians. We must pre-allocate the arrays
```julia
xdot = zeros(n)
Z = zeros(n,m)
```
Or create a partitioned vector and Jacobian for easy access to separate state and control jacobians
```julia
xdot = PartedVector(model)
Z = PartedMatrix(model)
```

Once the arrays are allocated, we can call using `evaluate!` and `jacobian!` (increments the evaluation count, recommended)
```julia
x,u = rand(x), rand(u)  # or some known test inputs
evaluate!(xdot,model,x,u)
jacobian!(Z,model,x,u)
```

If we created a partitioned Jacobian using `PartedMatrix(model)`, we can access the different pieces
```julia
fdx = Z.x   # ∂f/∂x
fdu = Z.u   # ∂f/∂u
fdt = Z.dt  # ∂f/∂dt
```

## API
### From Analytical Function
The following constructors can be used to create a model from an analytic function, with or without parameters or analyical Jacobians
```@docs
Model(f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}=Dict{Symbol,Any}())
Model(f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}())
Model(f::Function, ∇f::Function, n::Int64, m::Int64, p::NamedTuple, d::Dict{Symbol,Any}=Dict{Symbol,Any}())
Model(f::Function, ∇f::Function, n::Int64, m::Int64, d::Dict{Symbol,Any}=Dict{Symbol,Any}())
```

### From URDF
The following constructors can be used to create a Model from a URDF file
```@docs
Model(mech::Mechanism, torques::Array)
Model(mech::Mechanism)
Model(urdf::String)
Model(urdf::String,torques::Array{Float64,1})
```

### Evaluation methods
```@docs
evaluate!(ẋ::AbstractVector,model::Model{M,Continuous},x::AbstractVector,u::AbstractVector) where M <: ModelType
evaluate!(ẋ::AbstractVector,model::Model{M,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where {M <: ModelType,T}
jacobian!(Z::AbstractMatrix,model::Model{M,Continuous},x::AbstractVector,u::AbstractVector) where M <: ModelType
jacobian!(Z::AbstractArray{T},model::Model{M,Discrete},x::AbstractVector,u::AbstractVector,dt::T) where {M <: ModelType,T <: AbstractFloat}
jacobian!(Z::PartedMatTrajectory{T},model::Model{M,Discrete},X::VectorTrajectory{T},U::VectorTrajectory{T},dt::Vector{T}) where {M<:ModelType,T}
```

```@docs
evals(model::Model)
reset(model::Model)
```
