
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

function StaticPNSolver(prob::StaticProblem{L,T,<:StaticALObjective}, opts=StaticPNSolverOptions()) where {L,T}
    n,m,N = size(prob)
    NN = n*N + m*(N-1)
    stats = StaticPNStats()
    NP = n*N + sum(num_constraints(prob))

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

    dyn_inds, con_inds = gen_con_inds(prob.obj.constraints)

    # Set constant pieces of the Jacobian
    xinds,uinds = P.xinds, P.uinds
    ∇F[1] = Matrix(I,n,n+m+1)
    Ix = ∇F[1][:,xinds[1]]
    for k = 1:N-1
        D[dyn_inds[k+1], xinds[k+1]] .= -Ix
    end

    # Set constant elements of active set
    for ind in dyn_inds
        active_set[ind] .= true
    end

    StaticPNSolver(opts, stats, P, P̄, H, g, E, D, d, fVal, ∇F, active_set, dyn_inds, con_inds)
end

primals(solver::StaticPNSolver) = solver.P.Z

function update_constraints!(prob::StaticProblem, solver::DirectSolver, Z=prob.Z)
    conSet = get_constraints(prob)
    evaluate(conSet, Z)
    solver.fVal[1] = state(Z[1]) - prob.x0
    for k in 1:prob.N-1
        solver.fVal[k+1] = discrete_dynamics(prob.model, Z[k]) - state(Z[k+1])
    end
end

function update_active_set!(prob::StaticProblem, solver::StaticPNSolver, Z=prob.Z)
    conSet = get_constraints(prob)
    update_active_set!(conSet, Z, solver.opts.active_set_tolerance)
    for i = 1:length(conSet.constraints)
        copy_inds(solver.active_set, conSet.constraints[i].active, solver.con_inds[i])
    end
end

function constraint_jacobian!(prob::StaticProblem, solver::DirectSolver, Z=prob.Z)
    n,m,N = size(prob)
    conSet = get_constraints(prob)
    jacobian(conSet, Z)
    for k = 2:N
        solver.∇F[k] = discrete_jacobian(prob.model, Z[k-1])
    end
    return nothing
end

function constraint_jacobian_structure!(prob::StaticProblem, solver::DirectSolver, Z=prob.Z)
    n,m,N = size(prob)
    conSet = get_constraints(prob)
    jacobian(conSet, Z)
    for k = 1:N
        solver.∇F[k] =
        solver.∇F[k] = discrete_jacobian(prob.model, Z[k-1])
    end
    return nothing
end

function copy_inds(dest, src, inds)
    for i in eachindex(inds)
        dest[inds[i]] = src[i]
    end
end

function copy_jacobian!(D, con::ConstraintVals, cinds, xinds, uinds)
    N = length(xinds)
    if N in con.inds
        for (i,k) in enumerate(con.inds)
            D[cinds[i], xinds[k]] .= con.∇c[i]
        end
    else
        for (i,k) in enumerate(con.inds)
            zind = [xinds[k]; uinds[k]]
            D[cinds[i], zind] .= con.∇c[i]
        end
    end
end

function copy_constraints!(prob::StaticProblem, solver::DirectSolver, d=solver.d)
    conSet = get_constraints(prob)
    for (k,inds) in enumerate(solver.dyn_inds)
        d[inds] = solver.fVal[k]
    end
    for i = 1:length(conSet.constraints)
        copy_inds(d, conSet.constraints[i].vals, solver.con_inds[i])
    end
    return nothing
end

function copy_jacobians!(prob::StaticProblem, solver::DirectSolver, D=solver.D)
    n,m,N = size(prob)
    conSet = get_constraints(prob)
    xinds, uinds = solver.P.xinds, solver.P.uinds
    xi,ui = xinds[1], uinds[1]
    zi = [xi;ui]
    dinds = solver.dyn_inds
    cinds = solver.con_inds
    In = solver.∇F[1][:,xi]

    zind = [xinds[1]; uinds[1]]
    D[dinds[1], zind] .= solver.∇F[1][:,zi]
    for k = 1:N-1
        zind = [xinds[k]; uinds[k]]
        D[dinds[k+1], zind] .= solver.∇F[k+1][:,zi]
        # solver.D[dinds[k+1], xinds[k+1]] .= -In
    end

    for i = 1:length(conSet.constraints)
        copy_jacobian!(D, conSet.constraints[i], cinds[i], xinds, uinds)
    end
    return nothing
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
