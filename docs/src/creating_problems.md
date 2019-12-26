```@meta
CurrentModule = TrajectoryOptimization
```

# [4. Setting up a Problem](@id problem_section)
The [`Problem`](@ref) contains all of the information needed to solve a trajectory optimization
problem. At a minimum, this is the model, objective, and initial condition. A `Problem` is
passed to a solver, which extracts needed information, and may or may or not modify its
internal representation of the problem in order to solve it (e.g. the Augmented Lagrangian
solver combines the constraints and objective into a single Augmented Lagrangian objective.)
