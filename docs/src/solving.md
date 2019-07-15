# [6. Solving the Problem](@id solving_section)
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["solving.md"]
```

## Solving a problem
The default method for solving a problem is to specify both the problem and the solver, which are both modified in place: [`solve!(::Problem, ::AbstractSolver)`](@ref).

There are three additional methods defined for solving problems:
```@docs
solve!(::Problem, ::AbstractSolverOptions)
solve(::Problem, ::AbstractSolver)
solve(::Problem, ::AbstractSolverOptions)
```

The output during the solve can be controlled by toggling the `verbose` solver option. ALTRO, for example, has three verbosity flags:
```julia
opts = ALTROSolverOptions{Float64}()
opts.solver_al.verbose = true             # verbosity of Augmented Lagrangian
opts.solver_al.opts_uncon.verbose = true  # verbosity of iLQR
opts.solver_pn.verbose = true             # verbosity of projected newton
```
For more information on output and logging, see [Logging](@ref).

## Analyzing the Output

### Plotting
The primal solution variables are contained in [`Problem`](@ref). These can be plotted using [Plots.jl](https://github.com/JuliaPlots/Plots.jl)
```julia
plot(prob.X)
plot(prob.U)
```
which will plot all of the states or controls versus time step. Alternatively, you can plot a subset of them by passing in a `UnitRange`, for example:
```julia
plot(prob.X, 1:3)
```

### Stats
The statistics from the solve are stored in `solver.stats`. These stats are different for each solver, but typically include data such as runtime, iterations, final cost, and final constraint satisfaction.

Additionally, the following methods operate on the [`Problem`](@ref) and are often very useful
```@docs
cost(::Problem)
max_violation(prob::Problem)
```
