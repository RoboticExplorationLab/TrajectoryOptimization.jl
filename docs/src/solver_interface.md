```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["solver_interface.md"]
```

# Solver Interface
Solvers are currently organized into the following type tree:
* [`AbstractSolver`](@ref)
  * [`UnconstrainedSolver`](@ref)
  * [`ConstrainedSolver`](@ref)
    * [`DirectSolver`](@ref)


The solver options type is a lightweight container for all of the options the user can specify, such as tolerance values, printing verbosity (highly recommended), Boolean flags, etc. We highly suggest using [Parameters.jl](https://github.com/mauro3/Parameters.jl) to create this and easily specify the default options. All solver options should be mutable (e.g. `mutable struct NewSolverOptions{T} <: AbstractSolverOptions{T}`)

The solver type, on the other hand, is meant to contain all of the variables needed for the solve, including the model, objective, constraints, and other information originally in the `Problem`. This information is "duplicated" in the solver since oftentimes the solver with perform modifications to these when setting up the solve. For example, the [`AugmentedLagrangianSolver`](@ref) creates an `ALObjective` and uses that as it's objective instead. Similarly, ALTRO may convert the model to an [`InfeasibleModel`](@ref) to leverage an initial state trajectory. Therefore, once the solver is created, the problem is solved by simply calling `solve!(solver)`, which then runs the optimization.

## Defining a New Solver
For creating a new solver, e.g. `NewSolver`, the user must define two new types:
* `NewSolverOptions{T} <: AbstractSolverOptions{T}`
* `NewSolver{T} <: AbstractSolver{T}`

Below we list the methods needed to implement the different solver interfaces, along with a
list of inherited methods (that can be overloaded as needed). The docstrings for these functions
are listed in more detail down below.

### Unconstrained Solvers
Needed methods:
```julia
model = get_model(::NewSolver)
obj = get_objective(::NewSolver)
Z = get_trajectory(::NewSolver)
n,m,N = Base.size(::NewSolver)
x0 = get_initial_state(::NewSolver)
solve!(::NewSolver)
```

Needed fields (these will likely be replaced by getters in the near future):
* `opts` - instance of `AbstractSolverOptions`
* `stats` - mutable struct containing statistics on the solve

Inherited methods:
```julia
states(::NewSolver)
controls(::NewSolver)
initial_states!(::NewSolver, X0)
initial_controls!(::NewSolver, U0)
initial_trajectory!(::NewSolver, Z::Traj)
cost(::NewSolver, Z=get_trajectory(::NewSolver))
rollout!(::NewSolver)
```

### Constrained Solvers
Needed methods (in addition to those for Unconstrained Solvers):
```julia
get_constraints(::NewSolver)
```

Inherited methods (in addition to those for Unconstrained Solvers):
```julia
num_constraints(::NewSolver)
max_violation(::NewSolver, Z=get_trajectory(::NewSolver))
update_constraints!(::NewSolver, Z=get_trajectory(::NewSolver))
update_active_set!(::NewSolver, Z=get_trajectory(::NewSolver))
```

### Direct Solvers
Currently, direct solvers solve the problem by forming a large, sparse matrix by concatenating
the states and controls for all time steps. They need to generate a list of indices that map
the constraints to their location in the concatenated array of constraint values, which can
be done using [`gen_con_inds`](@ref). This must be stored as the `con_inds` field in the
`DirectSolver` (this will be replaced by a getter method in the near future). With this,
all `DirectSolver`s inherit the following methods for copying values to and from the large,
concatenated vectors and matrices:
```julia
copy_constraints!(d, ::DirectSolver)
copy_active_set!(a, ::DirectSolver)
copy_jacobians!(D, ::DirectSolver)
```

## Unconstrained Solvers

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

## Constrained Solvers

```@docs
ConstrainedSolver
update_constraints!
update_active_set!
constraint_jacobian!
```

## Direct Solvers
Direct solvers often perform similar operations, so the following methods are provided that should work with any direct solver

```@docs
DirectSolver
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
