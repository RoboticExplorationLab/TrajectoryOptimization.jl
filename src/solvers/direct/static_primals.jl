
struct StaticPrimals{T<:Real,N,M}
    Z::Vector{T}
    xinds::Vector{SVector{N,Int}}
    uinds::Vector{SVector{M,Int}}
    equal::Bool
end

function StaticPrimals(n::Int, m::Int, N::Int, equal=false)
    NN = n*N + m*(N-1) + equal*m
    Z = zeros(NN)
    uN = N-1 + equal

    xinds = [SVector{n}((n+m)*(k-1) .+ (1:n)) for k = 1:N]
    uinds = [SVector{m}(n + (n+m)*(k-1) .+ (1:m)) for k = 1:N]
    StaticPrimals(Z,xinds,uinds,equal)
end

function Base.copy(P::StaticPrimals)
    StaticPrimals(copy(P.Z),P.xinds,P.uinds,P.equal)
end

function Base.copyto!(P::StaticPrimals, Z::Traj)
    uN = P.equal ? length(Z) : length(Z)-1
    for k in 1:uN
        inds = [P.xinds[k]; P.uinds[k]]
        P.Z[inds] = Z[k].z
    end
    if !P.equal
        P.Z[P.xinds[end]] = state(Z[end])
    end
    return nothing
end

function Base.copyto!(V::AbstractVector{<:Real}, Z::Traj,
        xinds::Vector{<:AbstractVector}, uinds::Vector{<:AbstractVector})
    n,m,N = traj_size(Z)
    equal = (n+m)*N == length(V)

    uN = equal ? N : N-1
    for k in 1:uN
        inds = [xinds[k]; uinds[k]]
        V[inds] = Z[k].z
    end
    if !equal
        V[xinds[end]] = state(Z[end])
    end
    return nothing
end

function Base.copyto!(Z::Traj, P::StaticPrimals)
    uN = P.equal ? length(Z) : length(Z)-1
    for k in 1:uN
        inds = [P.xinds[k]; P.uinds[k]]
        Z[k].z = P.Z[inds]
    end
    if !P.equal
        xN = P.Z[P.xinds[end]]
        Z[end].z = [xN; control(Z[end])]
    end
    return nothing
end

function Base.copyto!(Z::Traj, V::Vector{<:Real},
        xinds::Vector{<:AbstractVector}, uinds::Vector{<:AbstractVector})
    n,m,N = traj_size(Z)
    equal = (n+m)*N == length(V)

    uN = equal ? N : N-1
    for k in 1:uN
        inds = [xinds[k]; uinds[k]]
        Z[k].z = V[inds]
    end
    if !equal
        xN = V[xinds[end]]
        Z[end].z = [xN; control(Z[end])]
    end
    return nothing
end


@with_kw mutable struct StaticPNStats{T}
    iterations::Int = 0
    c_max::Vector{T} = zeros(1)
    cost::Vector{T} = zeros(1)
end


@with_kw mutable struct StaticPNSolverOptions{T} <: DirectSolverOptions{T}
    verbose::Bool = true
    n_steps::Int = 1
    solve_type::Symbol = :feasible
    active_set_tolerance::T = 1e-3
    feasibility_tolerance::T = 1e-6
end



struct StaticPNSolver{T,N,M,NM,NNM,L1,L2,L3} <: DirectSolver{T}
    opts::StaticPNSolverOptions{T}
    stats::StaticPNStats{T}
    P::StaticPrimals{T,N,M}
    P̄::StaticPrimals{T,N,M}

    H::SparseMatrixCSC{T,Int}
    g::Vector{T}
    E::CostExpansion{T,N,M,L1,L2,L3}

    D::SparseMatrixCSC{T,Int}
    d::Vector{T}

    fVal::Vector{SVector{N,T}}
    ∇F::Vector{SMatrix{N,NM,T,NNM}}
    active_set::Vector{Bool}

    dyn_inds::Vector{SVector{N,Int}}
    con_inds::Vector{Vector{SV} where SV}
end

function StaticPNSolver(prob::StaticALProblem, opts=StaticPNSolverOptions())
    n,m,N = size(prob)
    NN = n*N + m*(N-1)
    stats = StaticPNStats()
    conSet = get_constraints(prob)
    NP = sum(num_constraints(prob))
    if !has_dynamics(conSet)
        NP += n*N
    end

    # Create concatenated primal vars
    P = StaticPrimals(n,m,N)
    P̄ = StaticPrimals(n,m,N)

    # Allocate Cost Hessian & Gradient
    H = spzeros(NN,NN)
    g = zeros(NN)
    E = CostExpansion(n,m,N)

    D = spzeros(NP,NN)
    d = zeros(NP)

    fVal = [@SVector zeros(n) for k = 1:N]
    ∇F = [@SMatrix zeros(n,n+m+1) for k = 1:N]
    active_set = zeros(Bool,NP)

    con_inds = gen_con_inds(conSet)

    # Set constant pieces of the Jacobian
    xinds,uinds = P.xinds, P.uinds

    dyn_inds = SVector{n,Int}[]
    StaticPNSolver(opts, stats, P, P̄, H, g, E, D, d, fVal, ∇F, active_set, dyn_inds, con_inds)
end

primals(solver::StaticPNSolver) = solver.P.Z
primal_partition(solver::StaticPNSolver) = solver.P.xinds, solver.P.uinds

function update_constraints!(prob::StaticProblem, solver::DirectSolver, Z=prob.Z)
    conSet = get_constraints(prob)
    evaluate!(conSet, Z)
end

function update_active_set!(prob::StaticProblem, solver::StaticPNSolver, Z=prob.Z)
    conSet = get_constraints(prob)
    update_active_set!(conSet, Z, Val(solver.opts.active_set_tolerance))
    for i = 1:length(conSet.constraints)
        copy_inds(solver.active_set, conSet.constraints[i].active, solver.con_inds[i])
    end
end

function constraint_jacobian!(prob::StaticProblem, solver::DirectSolver, Z=prob.Z)
    conSet = get_constraints(prob)
    jacobian!(conSet, Z)
    return nothing
end

function constraint_jacobian_structure(prob::StaticProblem, solver::DirectSolver, Z=prob.Z)
    n,m,N = size(prob)
    conSet = get_constraints(prob)
    idx = 0.0
    linds = jacobian_linear_inds(solver)

    NN = num_primals(solver)
    NP = num_duals(solver)
    D = spzeros(Int,NP,NN)

    # Number of elements in each block
    blk_len = map(con->length(con.∇c[1]), conSet.constraints)

    # Number of knot points for each constraint
    con_len = map(con->length(con.∇c), conSet.constraints)

    # Linear indices
    for (i,con) in enumerate(conSet.constraints)
        @show i
        for (j,k) in enumerate(con.inds)
            inds = idx .+ (1:blk_len[i])
            linds[i][j] = inds
            con.∇c[j] = inds
            idx += blk_len[i]
        end
    end
    copy_jacobians!(prob, solver, D)
    return D
end

function active_constraints(prob::StaticProblem, solver::StaticPNSolver)
    return solver.D[solver.active_set, :], solver.d[solver.active_set]  # this allocates
end


function cost_expansion!(prob::StaticALProblem, solver::StaticPNSolver)
    E = solver.E
    cost_expansion(E, prob.obj.obj, prob.Z)
    N = prob.N
    xinds, uinds = solver.P.xinds, solver.P.uinds
    H = solver.H
    g = solver.g

    for k = 1:N-1
        H[xinds[k],xinds[k]] .= E.xx[k]
        H[uinds[k],uinds[k]] .= E.uu[k]
        H[uinds[k],xinds[k]] .= E.ux[k]
        g[xinds[k]] .= E.x[k]
        g[uinds[k]] .= E.u[k]
    end
    H[xinds[N],xinds[N]] .= E.xx[N]
    g[xinds[N]] .= E.x[N]
    return nothing
end
