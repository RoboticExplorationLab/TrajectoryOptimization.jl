include("methods.jl")
include("models.jl")

function DI_objective_weights(model::Model, num_lift)
    n_lift, m_lift = model.n, model.m

    # objective
    q_lift = ones(n_lift)*1e-3
    Q_lift = [Diagonal(q_lift), Diagonal(q_lift), Diagonal(q_lift)]
    Qf_lift = [100.0*Diagonal(I,n_lift),100.0*Diagonal(I,n_lift),100.0*Diagonal(I,n_lift)]
    r_lift = ones(m_lift)*1e-1
    r_lift[4:6] .= 10
    R_lift = Diagonal(r_lift)

    return Q_lift, R_lift, Qf_lift
end
