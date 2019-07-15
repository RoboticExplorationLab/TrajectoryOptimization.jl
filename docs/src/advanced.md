# For Developers
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["advanced.md"]
```

## Logging

### Logging Levels
```@docs
OuterLoop
InnerLoop
InnerIters
```

### [`LogData`](@ref) Type
```@docs
LogData
cache_data!
cache_size
clear!(ldata::LogData)
add_col!
create_header
create_row
trim_entry
```

### [`SolverLogger`](@ref) Type
```@docs
SolverLogger
add_level!
print_header
print_row
Logging.handle_message(logger::SolverLogger, level, message::Symbol, _module, group, id, file, line; value=NaN, print=true, loc=-1, width=logger.default_width)
Logging.handle_message(logger::SolverLogger, level, message::String, _module, group, id, file, line; value=NaN)
```
