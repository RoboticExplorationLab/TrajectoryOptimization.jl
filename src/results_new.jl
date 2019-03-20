abstract type Results end

"$(TYPEDEF) Iterative LQR results"
struct iLQRResults <: Results
    X̄::Trajectory # states (n,N)
    Ū::Trajectory # controls (m,N-1)

    K::Trajectory  # State feedback gains (m,n,N-1)
    d::Trajectory  # Feedforward gains (m,N-1)

    S::Trajectory  # Cost-to-go Hessian (n,n,N)
    s::Trajectory  # Cost-to-go gradient (n,N)

    fdx::Trajectory # Discrete dynamics state jacobian (n,n,N-1)
    fdu::Trajectory # Discrete dynamics control jacobian (n,m,N-1)

    ρ::Vector{Float64} # Regularization
    dρ::Vector{Float64} # Regularization rate of change

    bp::BackwardPass

    function iLQRResults(X̄,Ū,K,d,S,s,fdx,fdu,ρ,dρ,bp)
        new(X̄,Ū,K,d,S,s,fdx,fdu,ρ,dρ,bp)
    end
end

function iLQRResults(n::Int,m::Int,N::Int)
    X̄  = [zeros(n)   for i = 1:N]
    Ū  = [zeros(m)   for i = 1:N-1]

    K  = [zeros(m,n) for i = 1:N-1]
    d  = [zeros(m)   for i = 1:N-1]

    S  = [zeros(n,n) for i = 1:N]
    s  = [zeros(n)   for i = 1:N]

    fdx = [zeros(n,n) for i = 1:N-1]
    fdu = [zeros(n,m) for i = 1:N-1]

    ρ = zeros(1)
    dρ = zeros(1)

    bp = BackwardPass(n,m,N)

    iLQRResults(X̄,Ū,K,d,S,s,fdx,fdu,ρ,dρ,bp)
end

function copy(r::iLQRResults)
    iLQRResults(copy(r.X̄),copy(r.Ū),copy(r.K),copy(r.d),copy(r.S),copy(r.s),copy(r.fdx),copy(r.fdu),copy(r.ρ),copy(r.dρ),copy(r.bp))
end

"$(TYPEDEF) Augmented Lagrangian results"
struct ALResults <: Results
    C::Trajectory      # Constraint values (p,N)
    C_prev::Trajectory # Previous constraint values (p,N)
    Cx::Trajectory     # State jacobian (n,n,N)
    Cu::Trajectory     # Control (k) jacobian (n,m,N-1)
    λ::Trajectory      # Lagrange multipliers (p,N)
    Iμ::Trajectory     # Penalty matrix (p,p,N)
    active_set::Trajectory # active set (p,N)
end

function ALResults(n::Int,m::Int,p::Int,N::Int,p_N::Int)
    C      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    C_prev = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    Cx     = [i != N ? zeros(p,n) : zeros(p_N,n)  for i = 1:N]
    Cu     = [i != N ? zeros(p,m) : zeros(p_N,0)  for i = 1:N-1]
    λ      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    Iμ     = [i != N ? Diagonal(ones(p)) : Diagonal(ones(p_N)) for i = 1:N]
    active_set = [i != N ? zeros(Bool,p) : zeros(Bool,p_N)  for i = 1:N]

    ALResults(C,C_prev,Cx,Cu,λ,Iμ,active_set)
end

function copy(r::ALResults)
    ALResults(deepcopy(r.C),deepcopy(r.C_prev),deepcopy(r.Cx),deepcopy(r.Cu),deepcopy(r.λ),deepcopy(r.Iμ),deepcopy(r.active_set))
end
