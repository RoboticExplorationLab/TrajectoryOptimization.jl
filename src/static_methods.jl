

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DYNAMICS FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


"Propagate the dynamics forward, storing the result in the next knot point"
function propagate_dynamics(::Type{Q}, model::AbstractModel, z_::KnotPoint, z::KnotPoint) where Q<:Implicit
    x_next = discrete_dynamics(Q, model, z)
    set_state!(z_, x_next)
end


"Evaluate the discrete dynamics for all knot points"
function discrete_dynamics!(f, model, Z::Traj)
    for k in eachindex(Z)
        f[k] = discrete_dynamics(model, Z[k])
    end
end

"Evaluate the discrete dynamics Jacobian for all knot points"
function discrete_jacobian!(∇f, model, Z::Traj)
    for k in eachindex(∇f)
        ∇f[k] = discrete_jacobian(model, Z[k])
    end
end
