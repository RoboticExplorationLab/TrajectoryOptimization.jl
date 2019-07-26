
max_con_viol = 1.0e-8
verbose=false

prob = copy(Problems.quad_obs)

# iLQR
T = Float64
opts_ilqr = iLQRSolverOptions{T}(verbose=true,
    iterations=300)
@time r0, s0 = solve(prob, opts_ilqr)

opts = StaticiLQRSolverOptions()
silqr = StaticiLQRSolver(prob, opts)



struct TestSolver{T,N,M,NM} <: AbstractSolver{T}
    opts::StaticiLQRSolverOptions{T}
    stats::Dict{Symbol,Any}

    # Data variables
    X̄::Vector{N} # states (n,N)
    Ū::Vector{M} # controls (m,N-1)

    K::Vector{NM}  # State feedback gains (m,n,N-1)
end
struct TestSolver3{T,N,M,NM,G,Q} <: AbstractSolver{T}
    opts::StaticiLQRSolverOptions{T}
    stats::Dict{Symbol,Any}

    # Data variables
    X̄::Vector{N} # states (n,N)
    Ū::Vector{M} # controls (m,N-1)

    K::Vector{NM}  # State feedback gains (m,n,N-1)
    d::Vector{M} # Feedforward gains (m,N-1)

    ∇F::Vector{G}# discrete dynamics jacobian (block) (n,n+m+1,N)

    S::Vector{Q} # Optimal cost-to-go expansion trajectory
    Q::Vector{Q} # cost-to-go expansion trajectory
end


# Init solver statistics
stats = Dict{Symbol,Any}(:timer=>TimerOutput())

# Init solver results
n = prob.model.n; m = prob.model.m; N = prob.N

X̄  = [@SVector zeros(T,n)   for k = 1:N]
Ū  = [@SVector zeros(T,m)   for k = 1:N-1]

K  = [@SMatrix zeros(T,m,n) for k = 1:N-1]
d  = [@SVector zeros(T,m)   for k = 1:N-1]

∇F = [@SMatrix zeros(T,n,n+m+1) for k = 1:N-1]

S = [@SMatrix zeros(T,n+m,n+m) for k = 1:N]
Q = [@SMatrix zeros(T,n+m,n+m) for k = 1:N]

ρ = zeros(T,1)
dρ = zeros(T,1)

TestSolver3(opts,stats,X̄,Ū,K,d,∇F,S,Q)
StaticiLQRSolver(opts,stats,X̄,Ū,K,d,∇F,S,Q,ρ,dρ)
