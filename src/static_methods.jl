

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DYNAMICS FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

"Evaluate the discrete dynamics for all knot points"
function discrete_dynamics!(f, model, Z::Traj)
    for k in eachindex(Z)
        f[k] = discrete_dynamics(model, Z[k])
    end
end

"Evaluate the discrete dynamics Jacobian for all knot points"
function discrete_jacobian!(∇f, model, Z::Traj)
    for k in eachindex(∇f)
        ∇f[k] .= discrete_jacobian(model, Z[k])
    end
end

function discrete_jacobian!(∇f::Vector{<:SMatrix}, model, Z::Traj)
    for k in eachindex(∇f)
        ∇f[k] = discrete_jacobian(model, Z[k])
    end
end
