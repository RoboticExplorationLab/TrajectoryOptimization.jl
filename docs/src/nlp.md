```@meta
CurrentModule = TrajectoryOptimization
```

# Converting to an NLP

```@contents
Pages = ["nlp.md"]
```

Trajectory optimization problems are really just nonlinear programs (NLPs). A handful of
high-quality NLP solvers exist, such as Ipopt, Snopt, and KNITRO. TrajectoryOptimization
provides an interface that allows methods that are amenable to use with a general-purpose
NLP solver. In the NLP, the states and constraints at every knot point are concatenated into
a single large vector of decision variables, and the cost hessian and constraint Jacobians
are represented as large, sparse matrices.

## Important Types
Below is the documentation for the types used to represent a trajectory optimization problem
as an NLP:

```@docs
NLPData
NLPConstraintSet
QuadraticViewCost
ViewKnotPoint
TrajData
NLPTraj
```

## The `TrajOptNLP` type
The most important type is the [`TrajOptNLP`](@ref), which is a single struct that has all
the required methods to evaluate the trajectory optimization problem as an NLP.

```@docs
TrajOptNLP
```

### Interface
Use the following methods on a `TrajOptNLP` `nlp`. Unless otherwise noted, `Z` is a single
vector of `NN` decision variables (where `NN` is the total number of states and controls across
all knot points).
```@docs
eval_f
grad_f!
hess_f!
hess_f_structure
eval_c!
jac_c!
jacobian_structure
hess_L!
```

The following methods are useful to getting important information that is typically required
by an NLP solver
```@docs
primal_bounds!
constraint_type
```

## MathOptInterface
The `TrajOptNLP` can be used to set up an `MathOptInterface.AbstractOptimizer` to solve
the trajectory optimization problem. For example if we want to use Ipopt and have already
set up our `TrajOptNLP`, we can solve it using `build_MOI!(nlp, optimizer)`:

```julia
using Ipopt
using MathOptInterface
nlp = TrajOptNLP(...)  # assume this is already set up
optimizer = Ipopt.Optimizer()
TrajectoryOptimization.build_MOI!(nlp, optimizer)
MathOptInterface.optimize!(optimizer)
```
