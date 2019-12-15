# [5. Pick a solver](@id solver_section)
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["solvers.md"]
```

## Creating a solver
Any of the following solvers can be created using the following method:

```julia
AbstractSolver(prob::Problem, opts::AbstractSolverOptions)
```
where the `opts` argument is the solver options for the desired solver. You can also call the solver constructor directly with the same arguments as above, e.g.
```julia
iLQRSolver(prob::Problem, opts::iLQRSolverOptions)
```
If the solver type is known beforehand, it is recommended to use the specific constructor rather than the `AbstractSolver` constructor, for clarity. In this case, the options argument is optional. The first is provided simply for generality and easy multiple dispatch.

All of the solver options types currently implemented use [Parameters.jl](https://github.com/mauro3/Parameters.jl), so can be initialized with the default constructor `<opts::AbstractSolverOptions>()`, e.g. `iLQRSolverOptions()`. The options can be set as keyword options in the constructor or specified afterwards.

## Solver Interface
Solvers are currently organized into the following type tree:
* [`AbstractSolver`](@ref)
  * [`UnconstrainedSolver`](@ref)
  * [`ConstrainedSolver`](@ref)
    * [`DirectSolver`](@ref)

For creating a new solver, e.g. `NewSolver`, the user must define two new types:
* `NewSolverOptions{T} <: AbstractSolverOptions{T}`
* `NewSolver{T} <: AbstractSolver{T}`

The solver options type is a lightweight container for all of the options the user can specify, such as tolerance values, printing verbosity (highly recommended), Boolean flags, etc. We highly suggest using [Parameters.jl](https://github.com/mauro3/Parameters.jl) to create this and easily specify the default options. All solver options should be mutable (e.g. `mutable struct NewSolverOptions{T} <: AbstractSolverOptions{T}`)

The solver type, on the other hand, is meant to contain all of the variables needed for the solve, including the model, objective, constraints, and other information originally in the `Problem`. This information is "duplicated" in the solver since oftentimes the solver with perform modifications to these when setting up the solve. For example, the [`AugmentedLagrangianSolver`](@ref) creates an `ALObjective` and uses that as it's objective instead. Similarly, ALTRO may convert the model to an [`InfeasibleModel`](@ref) to leverage an initial state trajectory. Therefore, once the solver is created, the problem is solved by simply calling `solve!(solver)`, which then runs the optimization.

The interfaces for the abstract solvers are described below:

### Unconstrained Solvers

```@docs
AbstractSolver
UnconstrainedSolver
states
controls
initial_trajectory!
initial_states!
initial_controls!
cost
cost_expansion!
```

### Constrained Solvers

```@docs
ConstrainedSolver
update_constraints!
update_active_set!
constraint_jacobian!
```

### Direct Solvers
Direct solvers often perform similar operations, so the following methods are provided that should work with any direct solver

```@docs
remove_bounds!
remove_constraint_type!
get_bounds
add_dynamics_constraints!
gen_con_inds
constraint_jacobian_structure
copy_constraints!
copy_active_set!
copy_jacobian!
copy_jacobians!
```

The solver must also contain the following fields:
* `opts`: Solver options for the solver (e.g. `opts::NewSolverOptions`)
* `stats::Dict{Symbol,Any}`: Dictionary containing pertinent statistics for the solve, such as run time, final max constraint violation, final cost, optimality criteria, number of iterations, etc.


# Implemented Solvers
## Iterative LQR (iLQR)
```@docs
iLQRSolver
iLQRSolverOptions
```

```@docs
AugmentedLagrangianSolver
AugmentedLagrangianSolverOptions
```


## ALTRO
```@docs
ALTROSolver
ALTROSolverOptions
```

## Direct Collocation (DIRCOL)
```@docs
DIRCOLSolver
DIRCOLSolverOptions
```

## Projected Newton
```@docs
ProjectedNewtonSolver
ProjectedNewtonSolverOptions
```
