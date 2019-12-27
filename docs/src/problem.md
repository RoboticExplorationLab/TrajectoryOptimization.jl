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
```@docs
change_integration
initial_controls!(::Problem, X0::Vector{<:AbstractVector})
initial_states!(::Problem, U0::Vector{<:AbstractVector})
Base.size(::Problem)
Base.copy(::Problem)
integration(::Problem)
states(::Problem)
controls(::Problem)
```
