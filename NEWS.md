# New in `v0.7`
Version `v0.7` is a dramatic reduction in scope from the previous versions.
The focus starting from this version is to provide a simple API for defining
trajectory optimization problems, leaving the task of evaluating and solving 
these to the solvers that wish to consume the `Problem` type.

Some notable changes:
- `Problem` constructor now has `x0` as position argument and `xf` as keyword argument
- The `Problem` type now stores a vector of `RobotDynamics.DiscreteDynamics` models,
allowing for changing dynamics models (i.e. hybrid dynamics) along the trajectory.
- Objectives no longer multiply "stage" costs by the time step automatically. This 
is left up to the user.
- The package no longer provides methods to evaluate cost or constraint expansions 
for the entire problem. Types such as `AbstractConstraintVals` or `CostExpansion`
have been removed.
- The package no longer provides native support for `MathOptInterface`. This 
functionality should be implemented by the solvers themselves.

# New in `v0.6`

## v`0.6.2`
Treats equality constraints as cones. 
Fixes deprecation warnings and small API changes from RobotDynamics v`0.4.3`, including:
- Replacing `Traj` with `SampledTrajectory`
- Using `RobotDynamics.dims` instead of `Base.size`
- Using `RobotDynamics.errstate_jacobian!` instead of `RobotDynamics.state_diff_jacobian!`

## Updated to new RobotDynamics `v0.4` API
Allows for both inplace and out-of-place dynamics, cost, and constraint evaluations.
Jacobians can be calculated using finite differences, forward AD, or user-specified.
Avoids prohibitively long compilation times for larger state and control dimensions.

- Replaced `length` and `size` methods with `state_dim`, `control_dim`, `output_dim` and 
`dims`. 
- `Problem` constructor is now of the form `Problem(model, obj, xf, tf, [x0, constraints])
Note that `x0` is now a required positional argument and `xf` is an optional keyword 
argument.
- Integration for dynamics is now specified by the model instead of a type parameter. 
- Support for direct collocation via MathOptInterface has been removed (will be implemented
in a separate repo in the future)


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