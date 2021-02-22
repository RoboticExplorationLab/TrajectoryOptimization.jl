# New in `v0.5`
## Support for Finite Differencing
Added support for finite differencing with `FiniteDiff` for dynamics, constraints, and cost functions.

## Added general nonlinear costs
Generic nonlinear costs are now officially supported and can be automatically differentiated using either ForwardDiff or FiniteDiff.

## Added generic `Expansion` type
The new `Expansion` type is provided for storing cost expansions, and is now preferred for use over `QuadraticCost`. It supports both `xx,xu,uu,x,u` and `Q,H,R,q,r` fields.
In general, the way cost functions were used has been cleaned up. `QuadraticObjective` and other such type aliases have been removed in favor of a less complicated API.

## Expanded Documentation
Documentation has been significantly updated.

# New in `v0.4`

## Conic Constraints
TrajectoryOptimization now add support for generalized inequalities / conic constraints. As of `0.4.0` only 2 cones are implemented:
* `NegativeOrthan` - equivalent to `Inequality`
* `SecondOrderCone`

Several constraints, most notably `NormConstraint` support passing in `SecondOrderCone` as the `sense` argument, which enforces that the output of 
the `evaluate` function lies in the second-order cone, with the last element 
of the vector being the scalar.

## Created the `AbstractConstraintValues` super-type
This allows solvers to define their own instantiations of this type, while providing many convenient methods automatically defined on the super-type.

## Added Cost Function for Rotations
Added `DiagonalLieCost` that penalizes the geodesic distance between a quaternion in the state and a reference quaternion.