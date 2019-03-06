
struct ADMMCost <: CostFunction
    costs::NamedTuple{M,NTuple{N,C}} where {M,N,C <: CostFunction}
    c::Function
    ∇c::Function
    q::Int            # Number of bodies
    b::Vector
    part_x::NamedTuple  # Partition for the state vectors
    part_u::NamedTuple  # Partition for the control vectors
end

function stage_cost(cost::ADMMCost, x::BlockVector, u::BlockVector)
    J = 0.0
    for body in keys(cost.costs)
        J += stage_cost(cost.costs[body],x[body],u[body])
    end
    return J
end

function taylor_expansion(cost::ADMMCost, x::BlockVector, u::BlockVector, body::Symbol)
    taylor_expansion(cost.costs[body],x[body],u[body])
end



struct ADMMResults #<: ConstrainedIterResults
    bodies::NTuple{N,Symbol} where N

    X::Trajectory  # States (n,N)
    U::Trajectory  # Controls (m,N)

    K::NamedTuple # Feedback (state) gain (m,n,N)
    d::NamedTuple  # Feedforward gain (m,N)

    X_::Trajectory # Predicted states (n,N)
    U_::Trajectory # Predicted controls (m,N)

    S::NamedTuple  # Cost-to-go hessian (n,n)
    s::NamedTuple  # Cost-to-go gradient (n,1)

    fdx::NamedTuple # State jacobian (n,n,N)
    fdu::NamedTuple # Control (k) jacobian (n,m,N-1)

    C::Trajectory      # Constraint values (p,N)
    C_prev::Trajectory # Previous constraint values (p,N)
    Iμ::Trajectory        # fcxtive constraint penalty matrix (p,p,N)
    λ::Trajectory # Lagrange multipliers (p,N)
    μ::Trajectory     # Penalty terms (p,N)

    Cx::Trajectory # State jacobian (n,n,N)
    Cu::Trajectory # Control (k) jacobian (n,m,N-1)
    #
    # t_prev::Trajectory
    # λ_prev::Trajectory
    #
    # nesterov::Vector{Float64}
    #
    # active_set::Trajectory
    #
    # ρ::Array{Float64,1}
    # dρ::Array{Float64,1}
    #
    # bp::BackwardPass
end


"""$(SIGNATURES) ADMM Results"""
function ADMMResults(bodies::NTuple{B,Symbol},ns::NTuple{B,Int},ms::NTuple{B,Int},p::Int,N::Int,p_N::Int) where B
    @assert length(bodies) == length(ns) == length(ms)
    part_x = create_partition(ns,bodies)
    part_u = create_partition(ms,bodies)
    n = sum(ns)
    m = sum(ms)

    X  = [BlockArray(zeros(n),part_x)   for i = 1:N]
    U  = [BlockArray(zeros(n),part_u)   for i = 1:N-1]

    K  = NamedTuple{bodies}([[zeros(m,n) for i = 1:N-1] for (n,m) in zip(ns,ms)])
    d  =  NamedTuple{bodies}([[zeros(m)   for i = 1:N-1] for m in ms])

    X_  = [BlockArray(zeros(n),part_x)   for i = 1:N]
    U_  = [BlockArray(zeros(m),part_u)   for i = 1:N-1]

    S  =  NamedTuple{bodies}([[zeros(n,n) for i = 1:N] for n in ns])
    s  =  NamedTuple{bodies}([[zeros(n)   for i = 1:N] for n in ns])


    fdx = NamedTuple{bodies}([[zeros(n,n) for i = 1:N-1] for n in ns])
    fdu = NamedTuple{bodies}([[zeros(n,m) for i = 1:N-1] for (n,m) in zip(ns,ms)])

    # Stage Constraints
    C      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    C_prev = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    Iμ     = [i != N ? Diagonal(ones(p)) : Diagonal(ones(p_N)) for i = 1:N]
    λ      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    μ      = [i != N ? ones(p) : ones(p_N)  for i = 1:N]

    Cx  = [i != N ? zeros(p,n) : zeros(p_N,n)  for i = 1:N]
    Cu  = [i != N ? zeros(p,m) : zeros(p_N,0)  for i = 1:N]

    t_prev      = [i != N ? ones(p) : ones(p_N)  for i = 1:N]
    λ_prev      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]

    nesterov = [0.;1.]

    active_set = [i != N ? zeros(Bool,p) : zeros(Bool,p_N)  for i = 1:N]

    ρ = zeros(1)
    dρ = zeros(1)

    bp = BackwardPass(n,m,N)

    ADMMResults(bodies,X,U,K,d,X_,U_,S,s,fdx,fdu,
        C,C_prev,Iμ,λ,μ,Cx,Cu)
    # ADMMResults(X,U,K,d,X_,U_,S,s,fdx,fdu,
    #     C,C_prev,Iμ,λ,μ,Cx,Cu,t_prev,λ_prev,nesterov,active_set,ρ,dρ,bp)
end
