"""
    dims(models::Vector{<:DiscreteDynamics})

Get the state and control dimensions for a vector of `N-1` dynamics models.
Returns `nx` and `nu`, vectors of length `N` corresponding to the state dimension 
and control dimension at each knot point, respectively. 

The last state dimension is calculated using `RobotDynamics.output_dim` of the last 
model. The last control dimension is assumed to be equal to that of the last model.

# Exceptions
Checks that the output dimensions and state dimensions of each dynamics model are 
consistent, and throws a `DimensionMismatch` otherwise.
"""
function RD.dims(models::Vector{<:DiscreteDynamics}) 
    nx = map(RD.state_dim, models)
    nu = map(RD.control_dim, models)
    push!(nx, RD.output_dim(models[end]))
    push!(nu, RD.control_dim(models[end]))
    for k = 1:length(models)
        ny = RD.output_dim(models[k])
        nx_next = nx[k+1]
        if nx_next != ny 
            throw(DimensionMismatch(
                "Model mismatch at time step $k.
                 Model $k has an output dimension of $ny but model $(k+1) has a state dimension of $nx_next.")
            )
        end
    end
    return nx, nu
end