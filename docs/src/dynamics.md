```@meta
CurrentModule = TrajectoryOptimization
```

# Evaluating the Dynamics and Jacobians
The dynamics are all specified by the `model::AbstractModel` using 
[RobotDynamics.jl](https://github.com/RoboticExplorationLab/RobotDynamics.jl).
We state here the methods that are implemented both in RobotDynamics and 
TrajectoryOptimization for working with dynamics and dynamics expansions.

To integrate the dynamics forward in time along an entire trajectory, use
```julia
rollout!(::Type{Q}, model, Z)
```
where `Q <: RobotDynamics.QuadratureRule` is any of the integration methods defined 
in RobotDynamics.

To evaluate the dynamics Jacobians, you can use `RobotDynamcis.DynamicsJacobian`
```julia
jac = RobotDynamics.DynamicsJacobian(model)
RobotDynamics.jacobian!(jac, model, z)                      # continuous Jacobian
RobotDynamics.discrete_jacobian!(::Type{Q}, jac, model, z)  # discrete Jacobian
``` 
which is a lightweight type that inherits from `AbstractArray`. You can extract 
the Jacobians with respect to the state and control by accessing the fields `A` and 
`B`, respectively, or using the more generic getter functions
```julia
RobotDynamics.get_A
RobotDynamics.get_B
```

Alternatively, TrajectoryOptimization.jl provides the more heavyweight 
`DynamicsExpansion` which can account for rotational states in a `LieGroupModel` 
(for more information see [Optimizing Rotations](@ref) or [RobotDynamics documentation](http://roboticexplorationlab.org/RobotDynamics.jl/dev/liemodel.html)).

```@docs
DynamicsExpansion
```

To evaluate the dynamics Jacobians for an entire trajectory, in the most general case, use
```julia
# Setup and initialization
N = 11    # number of knot points
dt = 0.1  # time step (sec)
n0,m = size(model)
n = state_diff_size(model)
G = [SizedMatrix{n0,n}(zeros(n0,n)) for k = 1:N]
D = [DynamicsExpansion{Float64}(n0,n,m)] for k = 1:N]
Z = Traj(n0,n,dt,N)
RobotDynamics.set_state!(Z[1], rand(model)[1])
set_controls!(Z, rand(m,N-1))
rollout!(RK4, model, Z)

TrajectoryOptimization.state_diff_jacobian!(G, model, Z)
TrajectoryOptimization.dynamics_expansion!(::Type{Q}, D::DynamicsExpansion, model, Z)
TrajectoryOptimization.error_expansion!(D::DynamicsExpansion, model, G)
```
For models that are not a `LieGroupModel`, the 3rd to last and last lines are not necessary.