using PartedArrays
n1,m1 = 4,2
n2,m2 = 4,0

Q1 = Diagonal(1I,n1)
R1 = Diagonal(1I,m1)
Qf1 = Diagonal(10I,n1)
Q2 = Diagonal(1I,n2)
R2 = Diagonal(1I,m2)
Qf2 = Diagonal(10I,n2)

cost1 = QuadraticCost(Q1,R1,zeros(m1,n1),zeros(n1),zeros(m1),0,Qf1,zeros(n1),0)
cost2 = QuadraticCost(Q2,R2,zeros(m2,n2),zeros(n2),zeros(m2),0,Qf2,zeros(n2),0)

bodies = (:a1,:m)
costs = NamedTuple{bodies}((cost1,cost2))
costs.a1

part_x = create_partition((n1,n2),bodies)
part_u = create_partition((m1,m2),bodies)
y0 = [0,1.,0,0]
v0 = zeros(m1)
z0 = [0.5,0,0,0]
w0 = zeros(m2)
x0 = [y0;z0]
d = 1
x = BlockArray(x0,part_x)
u = BlockArray(zeros(m1),part_u)

ϕ(x::BlockArray) = norm2(x.a1[1:2] - x.m[1:2]) - d^2
ϕ(x::Vector) = norm(x[part_x.a1][1:2] - x[part_x.m][1:2])^2 - d^2
function ∇ϕ(grad,x)
    y = x.a1[1:2]
    z = x.m[1:2]
    grad[1:2] = 2(y-z)
    grad[5:6] = -2(y-z)
    grad
end
∇ϕ(x) = begin grad = zeros(8); ∇ϕ(grad,x); grad end
function ∇ϕ(grad,x,b::Symbol)
    y = x.a1[1:2]
    z = x.m[1:2]
    if b == :a1
        grad[1:2] = 2(y-z)
    elseif b == :m
        grad[1:2] = -2(y-z)
    end
end
∇ϕ(x,b::Symbol) = begin grad = zeros(4); ∇ϕ(grad,x,b); grad end
ϕ(x)
ϕ(x0)
∇ϕ(x,:m)
ForwardDiff.gradient(ϕ,x0)

acost = ADMMCost(costs,ϕ,∇ϕ,2,[:a1],part_x,part_u)
stage_cost(acost,x,u)
stage_cost(cost1,y0,v0)
stage_cost(cost2,z0,w0)

taylor_expansion(acost,x,u,:m)
z0 == x.m
w0 == u.m

taylor_expansion(acost.costs.m,x.m,u.m)
part_x

ns = (n1,n2)
ms = (m1,m2)
p = 1
N = 11
ADMMResults(bodies,ns,ms,p,N,0);


X  = [BlockArray(zeros(sum(ns)),part_x)   for i = 1:N];
U  = [BlockArray(zeros(sum(ms)),part_u)   for i = 1:N-1];

K  = NamedTuple{bodies}([[zeros(m,n) for i = 1:N-1] for (n,m) in zip(ns,ms)])
d  =  NamedTuple{bodies}([[zeros(m)   for i = 1:N-1] for m in ms])

testres(X,U,K,d);


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


struct ADMMResults2
    results::Vector{ConstrainedVectorResults}
    b::Int
end

struct ADMMResults #<: ConstrainedIterResults
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

    ADMMResults(X,U,K,d,X_,U_,S,s,fdx,fdu.
        C,C_prev,Iμ,λ,μ,Cx,Cu)
    # ADMMResults(X,U,K,d,X_,U_,S,s,fdx,fdu,
    #     C,C_prev,Iμ,λ,μ,Cx,Cu,t_prev,λ_prev,nesterov,active_set,ρ,dρ,bp)
end
