```@meta
CurrentModule = TrajectoryOptimization
```

# [Problem](@id problem_api)

```@contents
Pages = ["problem.md"]
```

## Definition
```@docs
Problem
```

## Methods

### Getters
```@docs
cost(::Problem)
states(::Problem)
controls(::Problem)
horizonlength
get_objective
get_constraints
get_model
get_trajectory
get_initial_time
get_final_time
get_initial_state
get_final_state
is_constrained
gettimes
```

### Setters
```@docs
initial_controls!(::Problem, X0::Vector{<:AbstractVector})
initial_states!(::Problem, U0::Vector{<:AbstractVector})
initial_trajectory!
set_initial_state!
set_goal_state!
```

### Other
```@docs
rollout!
```