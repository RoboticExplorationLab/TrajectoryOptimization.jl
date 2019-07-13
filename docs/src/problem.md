# [3. Setting up a Problem](@id problem_section)
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["problem.md"]
```

## Creating a Problem
```@docs
Problem
```

## Methods
```@docs
update_problem
initial_controls!
initial_states!
set_x0!
Base.size(::Problem)
Base.copy(::Problem)
is_constrained
max_violation(::Problem{T}) where T
final_time
```
