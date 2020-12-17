```@meta
CurrentModule = TrajectoryOptimization
```

# Optimizing Rotations
Optimization over the space of rotations is non-trivial due to the group structure 
of 3D rotations. TrajectoryOptimization.jl provides methods for accounting for this
group structure, both in the constraints and in the objective. We make the assumption
that 3D rotation only show up in the state vector, and never in the control vector.
TrajectoryOptimization.jl relies on the dynamics model to determine if the state
vector contains rotations. See the 
[RobotDynamics.jl documentation](http://roboticexplorationlab.org/RobotDynamics.jl/dev/liemodel.html)
for more details on defining models with rotations. From here, we assume that we are
dealing with a model that inherits from [`RobotDynamics.LieGroupModel`](http://roboticexplorationlab.org/RobotDynamics.jl/dev/liemodel.html#LieGroupModel-API-1).

## Cost Functions (experimental)
While normal quadratic cost functions can work with rotations (e.g. ``q_k^T Q q_k``, where ``q_k`` is a quaternion, MRP, or RP), this distance metric isn't well-defined. 
Since we often want to penalize the distance from a reference rotation 
(such as a nominal or goal state), TrajectoryOptimization.jl provides a couple
different methods for penalizing distance from a reference rotation. However, we've 
discovered that the quaternion geodesic distance: 
```math
\min 1 \pm q_d^T q_k
```
where ``q_d`` is the desired, or reference, quaternion, works the best. We've also 
found that, while technically incorrect, a naive quadratic penalty can work quite
well, especially when the difference between the rotations isn't significant. 

The following cost functions are provided. Note that these methods should still be 
considered experimental, and the interface made change in the future. If you encounter
any issues using these functions, please submit an issue.

```@docs
DiagonalQuatCost
QuatLQRCost
ErrorQuadratic
```

## Cost and Constraint Expansions 
The key to performing optimization on Lie Groups such as rotations is to perform 
the optimization on the tangent bundle, or simply the hyperplane tangent to the 
group at the current iterate. Since the tangent space is Euclidean, we use 
standard optimization methods such as Newton's method to find a step direction. 
After we find a candidate step direction, we project back onto the group using 
the exponential map (or any other "retraction map"). 
For rotations and unit quaternions, this means we 
can either use the exponential map, or use the conversions between unit quaternions
and MRPs or RPs (Rodrigues Parameters). Since most optimization methods require
gradient or Hessian information, we need to correctly account for the mapping to the
the tangent plane when computing derivatives.

TrajectoryOptimization.jl handles this by first computing the derivatives as normal, 
treating the state (naively) as a vector in Euclidean space. This means methods such 
as ForwardDiff can be used without problem. These derivatives are then "converted" 
to work on the error state. This conversion ends up being one or two extra matrix 
multiplications with the so-called "error-state Jacobian," which is a function of the
rotation at the current iterate. 

Therefore, computing the full derivative information is split into 2 steps:
1. Compute the derivatives as normal.
2. Compute the "error expansion," using the result of step 1.

For objectives this looks like:
```julia
TrajectoryOptimization.state_diff_jacobian!(G, model, Z)
TrajectoryOptimization.cost_expansion!(E0, obj, Z, [init, rezero])
TrajectoryOptimization.error_expansion!(E, E0, model, Z, G)
```
The corrected expansion is stored in `E::QuadraticObjective`. The first line computes
the error-state Jacobians, storing them in `G`. The intermediate expansion is stored 
in `E0::QuadraticObjective`. For models that do not have rotations, `E === E0`
and the last line is a no-op.

For dynamics this looks very similar:
```julia
TrajectoryOptimization.state_diff_jacobian!(G, model, Z)
TrajectoryOptimization.dynamics_expansion!(::Type{Q}, D::DynamicsExpansion, model, Z)
TrajectoryOptimization.error_expansion!(D::DynamicsExpansion, model, G)
```
The first line can be omitted if it has already be computed for the current 
trajectory `Z::AbstractTrajectory`.

This functionality is still under development for constraints. Since augmented 
Lagrangian methods incorporate the constraints into the objective, the error 
expansion is computed on the entire cost, rather than computing an intermediate
error expansion for the constraints, saving computation. If you need this 
functionality, please submit an issue.

The experimental interface is currently:
```julia
TrajectoryOptimization.jacobian!(conSet::AbstractConstraintSet, Z::AbstractTrajectory, [init])
TrajectoryOptimization.error_expansion!(conSet::AbstractConstraintSet, model, G)
```