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
