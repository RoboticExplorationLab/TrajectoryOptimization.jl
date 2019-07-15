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
If the solver type is known beforehand, it is recommended to use the specific constructor rather than the `AbstractSolver` constructor, for clarity. The first is provided simply for generality and easy multiple dispatch.

All of the solver options types currently implemented use [Parameters.jl](https://github.com/mauro3/Parameters.jl), so can be initialized with the default constructor `<opts::AbstractSolverOptions>{T}()`, e.g. `iLQRSolverOptions{Float64}()`. The options can be set as keyword options in the constructor or specified afterwards.


## Solver Interface
For creating a new solver, e.g. `NewSolver`, the user must define two new types:
* `NewSolverOptions{T} <: AbstractSolverOptions{T}`
* `NewSolver{T} <: AbstractSolver{T}`

The solver options type is a lightweight container for all of the options the user can specify, such as tolerance values, printing verbosity (highly recommended), Boolean flags, etc. We highly suggest using [Parameters.jl](https://github.com/mauro3/Parameters.jl) to create this and easily specify the default options. All solver options should be mutable (e.g. `mutable struct NewSolverOptions{T} <: AbstractSolverOptions{T}`)

The solver type is meant to contain all of the variables needed for the solve, which should be initialized once when the solver is created and then modified in place. These could be values such as dual variables (Lagrange multipliers), constraint values, Jacobians, etc. The solver must define the following methods:

```@docs
solve!(prob::Problem, solver::AbstractSolver)
AbstractSolver(::Problem, opts::AbstractSolverOptions)
```
The following methods are optional, but recommended
```@docs
copy(::AbstractSolver)
reset!(::AbstractSolver)
size(::AbstractSolver)
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

## Augmented Lagrangian
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
