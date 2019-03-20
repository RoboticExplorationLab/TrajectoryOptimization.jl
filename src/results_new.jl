abstract type Results{T<:AbstractFloat} end

"$(TYPEDEF) Iterative LQR results"
struct iLQRResults{T} <: Results{T}
    X̄::VectorTrajectory{T} # states (n,N)
    Ū::VectorTrajectory{T} # controls (m,N-1)

    K::MatrixTrajectory{T}  # State feedback gains (m,n,N-1)
    d::VectorTrajectory{T}  # Feedforward gains (m,N-1)

    S::MatrixTrajectory{T}  # Cost-to-go Hessian (n,n,N)
    s::VectorTrajectory{T}  # Cost-to-go gradient (n,N)

    ∇F::MatrixTrajectory{T} # discrete dynamics jacobian (block) (n,n+m+1,N)

    ρ::Vector{T} # Regularization
    dρ::Vector{T} # Regularization rate of change

    bp::BackwardPass{T}
end

function iLQRResults(p::Problem{T}) where T
    n = p.model.n; m = p.model.m; N = p.N

    X̄  = [zeros(T,n)   for i = 1:N]
    Ū  = [zeros(T,m)   for i = 1:N-1]

    K  = [zeros(T,m,n) for i = 1:N-1]
    d  = [zeros(T,m)   for i = 1:N-1]

    S  = [zeros(T,n,n) for i = 1:N]
    s  = [zeros(T,n)   for i = 1:N]

    ∇F = [zeros(T,n,n+m+1) for i = 1:N-1]

    ρ = zeros(T,1)
    dρ = zeros(T,1)

    bp = BackwardPass{T}(n,m,N)

    iLQRResults{T}(X̄,Ū,K,d,S,s,∇F,ρ,dρ,bp)
end

function copy(r::iLQRResults{T}) where T
    iLQRResults{T}(copy(r.X̄),copy(r.Ū),copy(r.K),copy(r.d),copy(r.S),copy(r.s),copy(r.∇F),copy(r.ρ),copy(r.dρ),copy(r.bp))
end


struct BackwardPassNew{T<:AbstractFloat}
    Qx::VectorTrajectory{T}
    Qu::VectorTrajectory{T}
    Qxx::MatrixTrajectory{T}
    Qux::MatrixTrajectory{T}
    Quu::MatrixTrajectory{T}
    Qux_reg::MatrixTrajectory{T}
    Quu_reg::MatrixTrajectory{T}
end

function BackwardPassNew(p::Problem{T}) where T
    Qx = [zeros(T,n) for i = 1:N-1]
    Qu = [zeros(T,m) for i = 1:N-1]
    Qxx = [zeros(T,n,n) for i = 1:N-1]
    Qux = [zeros(T,m,n) for i = 1:N-1]
    Quu = [zeros(T,m,m) for i = 1:N-1]

    Qux_reg = [zeros(T,m,n) for i = 1:N-1]
    Quu_reg = [zeros(T,m,m) for i = 1:N-1]

    BackwardPassNew{T}(Qx,Qu,Qxx,Qux,Quu,Qux_reg,Quu_reg)
end

function copy(bp::BackwardPassNew{T}) where T
    BackwardPassNew{T}(deepcopy(bp.Qx),deepcopy(bp.Qu),deepcopy(bp.Qxx),deepcopy(bp.Qux),deepcopy(bp.Quu),deepcopy(bp.Qux_reg),deepcopy(bp.Quu_reg))
end

function reset(bp::BackwardPass)
    N_ = length(bp.Qx)
    for k = 1:N-1
        Qx[k] = zero(Qx[k]); Qu[k] = zero(Qu[k]); Qxx[k] = zero(Qxx[k]); Quu[k] = zero(Quu[k]); Qux[k] = zero(Qux[k])
        Quu_reg[k] = zero(Quu_reg[k]); Qux_reg[k] = zero(Qux_reg[k])
    end
end


"$(TYPEDEF) Augmented Lagrangian results"
struct ALResults <: Results{T<:Real}
    C::VectorTrajectory{T}      # Constraint values [(p,N-1) (p_N)]
    C_prev::VectorTrajectory{T} # Previous constraint values [(p,N-1) (p_N)]
    ∇C::MatrixTrajectory{T}   # Constraint jacobians [(p,n+m,N-1) (p_N,n)]
    λ::VectorTrajectory{T}      # Lagrange multipliers [(p,N-1) (p_N)]
    Iμ::DiagonalTrajectory{T}     # Penalty matrix [(p,p,N-1) (p_N,p_N)]
    active_set::VectorTrajectory{Bool} # active set [(p,N-1) (p_N)]
end

function ALResults(prob::Problem{T}) where T
    n = prob.model.n; m = prob.model.m; N = prob.N
    p = num_stage_constraints(prob.constraints)
    p_N = num_terminal_constraints(prob.constraints)

    C      = [i != N ? zeros(T,p) : zeros(T,p_N)  for i = 1:N]
    C_prev = [i != N ? zeros(T,p) : zeros(T,p_N)  for i = 1:N]
    ∇C     = [i != N ? zeros(T,p,n+m) : zeros(T,p_N,n)  for i = 1:N]
    λ      = [i != N ? zeros(T,p) : zeros(T,p_N)  for i = 1:N]
    Iμ     = [i != N ? Diagonal(ones(T,p)) : Diagonal(ones(T,p_N)) for i = 1:N]
    active_set = [i != N ? zeros(Bool,p) : zeros(Bool,p_N)  for i = 1:N]

    ALResults{T}(C,C_prev,∇C,λ,Iμ,active_set)
end

function copy(r::ALResults{T}) where T
    ALResults{T}(deepcopy(r.C),deepcopy(r.C_prev),deepcopy(r.∇C),deepcopy(r.λ),deepcopy(r.Iμ),deepcopy(r.active_set))
end
