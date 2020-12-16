```@meta
CurrentModule = TrajectoryOptimization
```

# Problem

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
Base.size(::Problem)
num_constraints(::Problem)
get_constraints
get_model
get_trajectory
get_initial_state
RobotDynamics.get_times(::Problem)
integration(::Problem)
is_constrained
```

### Setters
```@docs
initial_controls!(::Problem, X0::Vector{<:AbstractVector})
initial_states!(::Problem, U0::Vector{<:AbstractVector})
initial_trajectory!
set_initial_state!(::Problem, x0::AbstractVector)
set_initial_time!(::Problem, t0::Real)
set_goal_state!
change_integration
```

### Other
```@docs
rollout!
Base.copy(::Problem)
```