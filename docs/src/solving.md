```@meta
CurrentModule = TrajectoryOptimization
```

# [5. Solving the Problem](@id solving_section)
Solving a trajectory optimization is as simple as picking a solver and calling
the `solve!` method. There are several solvers implemented in TrajectoryOptimization.jl, and each has its own limitations and solver options.
See [Solvers](@ref) section for a list of the currently implemented solvers, their options, and their associated docstrings. See the [Solver Interface](@ref) section for more information on addition new solvers or leveraging the generic methods implemented by each solver.

## Creating a Solver
Every solver is initialized with, at a minimum, a `Problem` and (optionally) the associated `SolverOptions` type. For example, let's say we want to solve our cartpole problem created in the previous section with AL-iLQR. Since iLQR is the default unconstrained solver for the Augmented Lagrangian solver, this is as easy as:
```julia
solver = AugmentedLagrangianSolver(prob)
```

## Modifying solver options
We have a couple options for changing solver options. First, we can create the `SolverOptions` type, modify the desired fields, and then pass it into the constructor for the solver. For iLQR this looks like:
```julia
opts_ilqr = iLQRSolverOptions()                     # all fields default
opts_ilqr = iLQRSolverOptions(cost_tolerance=1e-4)  # specify options by keyword
opts_ilqr.iterations = 50                           # set fields manually
solver = iLQRSolver(prob, opts_ilqr)                # create the solver with the given options
```

Second, we can create the solver, and then modify it's options:
```julia
solver = iLQRSolver(prob)
solver.opts.iterations = 50
```
It should be noted that right now, the solver does NOT create a copy of the solver options when it's created. That is, changing `opts_ilqr` after creating the solver will change the options inside of `solver`, even after it's created, and vice versa.

### Nested Solver Options
Some solvers, such as [`AugmentedLagrangianSolver`](@ref) and [`ALTROSolver`](@ref) rely directly on other solvers. As such, their solver option types store the solver options of their dependent solvers explicitly:
```julia
opts_al = AugmentedLagrangianSolverOptions()
opts_al.opts_uncon.iterations = 50  # set iLQR iterations
opts_al.iterations = 20             # set outer loop iterations
```

## Solving the Problem
Solving the problem is as simple as calling `solve!` on the solver:
```julia
solver = AugmentedLagrangianSolver(prob)
solve!(solver)
```

## Analyzing the Solution
There a few different ways to analyze the solution after the solve, detailed in the following
sections.

### Getting the solution
The state and control trajectories can be extracted using the following commands:
```julia
states(solver::AbstractSolver)
controls(solver::AbstractSolver)
```
These can be converted to 2D arrays with simple horizontal concatenation: `hcat(X...)`.
Note that these methods work on `Problem` types as well.

### Logging
All solvers support some type of output to `stdout` during the solve. This is controlled
with the `verbose` solver option.

### Statistics
Each solver records a log of certain stats during the solve. They are located in `solver.stats`,
and include things such as cost per iteration, constraint violation per iteration, cost
decrease, etc.

### Plotting
TrajectoryOptimization.jl currently does not have any plotting recipes defined, although this
is on the list of feature we are going to be adding.
