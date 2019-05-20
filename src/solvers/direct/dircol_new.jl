cost(prob::Problem, solver::DIRCOLSolver) = cost(prob, solver.Z)

cost(prob::Problem, Z::Primals) = cost(prob.obj, Z.X, Z.U)

"""Number of collocation constraints"""
num_colloc(prob::Problem)::Int = (prob.N-1)*prob.model.n


function convertInf!(A::VecOrMat{Float64},infbnd=1.1e20)
    infs = isinf.(A)
    A[infs] = sign.(A[infs])*infbnd
    return nothing
end

function cost_gradient!(grad_f, prob::Problem, solver::DirectSolver, Z::Primals=solver.Z)
    n,m,N = size(prob)
    grad = reshape(grad_f, n+m, N)
    part = (x=1:n, u=n+1:n+m)
    X,U = Z.X, Z.U
    for k = 1:N-1
        grad_k = PartedVector(view(grad,:,k), part)
        gradient!(grad_k, prob.obj[k], X[k], U[k])
        grad_k ./= (N-1)
    end
    grad_k = PartedVector(view(grad,1:n,N), part)
    gradient!(grad_k, prob.obj[N], X[N])
    return nothing
end

function cost_gradient!(grad_f, prob::Problem, X::AbstractVectorTrajectory, U::AbstractVectorTrajectory)
    n,m,N = size(prob)
    grad = reshape(grad_f, n+m, N)
    part = (x=1:n, u=n+1:n+m)
    for k = 1:N-1
        grad_k = PartedVector(view(grad,:,k), part)
        gradient!(grad_k, prob.obj[k], X[k], U[k])
        grad_k ./= (N-1)
    end
    grad_k = PartedVector(view(grad,1:n,N), part)
    gradient!(grad_k, prob.obj[N], X[N])
    return nothing
end

function traj_points!(prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}, Z=solver.Z) where T
    n,m,N = size(prob)
    dt = prob.dt
    Xm = solver.X_
    fVal = solver.fVal
    X,U = Z.X, Z.U
    for k = 1:N-1
        Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
    end
    return Xm
end

function traj_points(prob::Problem, X::AbstractVectorTrajectory{T}, U, fVal) where T
    n,m,N = size(prob)
    dt = prob.dt
    Xm = [zeros(T,n) for k = 1:N-1]
    for k = 1:N-1
        Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
    end
    return Xm
end

function TrajectoryOptimization.dynamics!(prob::Problem{T,Continuous}, solver::DirectSolver, Z=solver.Z) where T<:AbstractFloat
    for k = 1:prob.N
        evaluate!(solver.fVal[k], prob.model, Z.X[k], Z.U[k])
    end
end


function TrajectoryOptimization.dynamics!(prob::Problem{T,Discrete}, solver::DirectSolver, Z=solver.Z) where T
    for k = 1:prob.N
        evaluate!(solver.fVal[k], prob.model, Z.X[k], Z.U[k], prob.dt)
    end
end

function TrajectoryOptimization.dynamics(prob::Problem{T,Continuous}, X, U) where T<:AbstractFloat
    n,m,N = size(prob)
    fVal = [zeros(eltype(X[1]),n) for k = 1:N]
    for k = 1:prob.N
        evaluate!(fVal[k], prob.model, X[k], U[k])
    end
    return fVal
end

function dynamics_jacobian(prob::Problem, X::AbstractVectorTrajectory{T}, U) where T
    n,m,N = size(prob)
    part_f = create_partition2(prob.model)
    ∇F         = [PartedMatrix(zeros(T,n,n+m),part_f)           for k = 1:N]
    for k = 1:prob.N
        jacobian!(∇F[k], prob.model, X[k], U[k])
    end
    return ∇F
end

function constraint_jacobians(prob::Problem, X::AbstractVectorTrajectory, U::AbstractVectorTrajectory)
    n,m,N = size(prob)
    ∇C         = [PartedMatrix(prob.constraints[k],n,m,:stage) for k = 1:N-1]
    ∇C         = [∇C..., PartedMatrix(prob.constraints[N],n,m,:terminal)]
    for k = 1:N
        if k == prob.N
            jacobian!(∇C[k], prob.constraints[k], X[k])
        else
            jacobian!(∇C[k], prob.constraints[k], X[k], U[k])
        end
    end
    return ∇C
end

function calculate_jacobians!(prob::Problem, solver::DirectSolver, Z=solver.Z)
    for k = 1:prob.N
        jacobian!(solver.∇F[k], prob.model, Z.X[k], Z.U[k])
        if k == prob.N
            jacobian!(solver.∇C[k], prob.constraints[k], Z.X[k])
        else
            jacobian!(solver.∇C[k], prob.constraints[k], Z.X[k], Z.U[k])
        end
    end
end


# function update_constraints!(g, prob::Problem, Z)
#     n,m,N = size(prob)
#     p,pN = num_stage_constraints(prob), num_terminal_constraints(prob)
#     P = (N-1)*p
#     X,U = Z.X, Z.U
#
#     g_stage = reshape(view(g,1:P), p, N-1)
#     g_term = view(g,P+1:length(g))
#
#     for k = 1:N-1
#         evaluate!(g_stage[:,k], prob.constraints, X[k], U[k])
#     end
#     evaluate!(g_term, prob.constraints, X[N])
# end

function update_constraints!(g, prob::Problem, solver::DIRCOLSolver, X, U) where T
    n,m,N = size(prob)
    p = solver.p
    pcum = [0; cumsum(p)]
    for k = 1:N
        if p[k] > 0
            k == N ? part = :terminal : part = :stage
            part_c = solver.C[k].parts
            inds = pcum[k] .+ (1:p[k])
            if k == N
                evaluate!(PartedArray(view(g, inds), part_c), prob.constraints[k], X[k])
            else
                evaluate!(PartedArray(view(g, inds), part_c), prob.constraints[k], X[k], U[k])
            end
        end
    end
end

function update_constraints!(g, prob::Problem, X::AbstractVectorTrajectory, U::AbstractVectorTrajectory) where T
    n,m,N = size(prob)
    p_colloc = n*(N-1)
    g_colloc = view(g,1:p_colloc)
    collocation_constraints!(g_colloc, prob, X, U)

    g_custom = view(g, p_colloc + 1:length(g))
    p = num_constraints(prob.constraints)
    offset = 0
    C          = [PartedVector(prob.constraints[k],:stage)     for k = 1:N-1]
    C          = [C...,  PartedVector(prob.constraints[N],:terminal)]
    for k = 1:N
        c = PartedVector(view(g_custom, offset .+ (1:p[k])), C[k].parts)
        if k == N
            evaluate!(c, prob.constraints[k], X[k])
        else
            evaluate!(c, prob.constraints[k], X[k], U[k])
        end
        offset += p[k]
    end
end

function partition_constraint_jacobian(jac::AbstractMatrix, prob::Problem)
    n,m,N = size(prob)
    p_colloc = (N-1)*n
    jac_colloc = view(jac, 1:p_colloc, :)
    jac_custom = view(jac, p_colloc+1:size(jac,1), :)
    return jac_colloc, jac_custom
end

function partition_constraint_jacobian(jac::AbstractVector, prob::Problem)
    n,m,N = size(prob)
    p_colloc = num_colloc(prob)
    jac_colloc = view(jac, 1:p_colloc*2(n+m))
    jac_custom = view(jac, p_colloc*2(n+m)+1:length(jac))
    return jac_colloc, jac_custom
end

function constraint_jacobian!(jac, prob::Problem, X::AbstractVectorTrajectory, U::AbstractVectorTrajectory)
    n,m,N = size(prob)
    p_colloc = num_colloc(prob)
    jac_colloc, jac_custom = partition_constraint_jacobian(jac, prob)
    collocation_constraint_jacobian!(jac_colloc, prob, X, U)
    p = num_constraints(prob)

    ∇C = constraint_jacobians(prob, X, U)

    off1 = p_colloc
    off2 = 0
    ns = p[1:N-1]
    ms = ones(Int,N-1)*(n+m)
    b1 = p
    b2 = ones(Int,N)*(n+m)
    b2[N] = n

    for k = 1:N
        block = get_jac_block(jac_custom, k, b1, b2, ns, ms)
        block .= ∇C[k]
        off1 += p[k]
        off2 += n+m
    end
end

function constraint_jacobian!(jac, prob::Problem, solver::DirectSolver, X, U)
    n,m,N = size(prob)
    p_colloc = num_colloc(prob)
    p = solver.p
    off = 0
    for k = 1:N
        part_c = solver.∇C[k].parts
        if k == N
            n_blk = p[k]*n
            block = PartedArray(reshape(view(jac, off .+ (1:n_blk)), p[k], n), part_c)
            jacobian!(block, prob.constraints[k], X[k])
        else
            n_blk = p[k]*(n+m)
            block = PartedArray(reshape(view(jac, off .+ (1:n_blk)), p[k], n+m), part_c)
            jacobian!(block, prob.constraints[k], X[k], U[k])
        end
        off += n_blk
    end
end

function constraint_jacobian_sparsity!(jac::AbstractMatrix, prob::Problem)
    n,m,N = size(prob)
    p_colloc = n*(N-1)
    jac_colloc, jac_custom = partition_constraint_jacobian(jac, prob)
    collocation_constraint_jacobian_sparsity!(jac_colloc, prob)

    p = num_constraints(prob)
    off = p_colloc*2(n+m)
    ns = p[1:N-1]
    ms = ones(Int,N-1)*(n+m)
    b1 = p
    b2 = ones(Int,N)*(n+m)
    b2[N] = n
    for k = 1:N
        n_blk = p[k]*b2[k]
        block = get_jac_block(jac_custom, k, b1, b2, ns, ms)
        block .= reshape(off .+ (1:n_blk), p[k], b2[k])
        off += n_blk
    end
end

function collocation_constraints!(g, prob::Problem, X::AbstractVectorTrajectory, U::AbstractVectorTrajectory) where T
    n,m,N = size(prob)
    @assert isodd(N)
    dt = prob.dt

    # Reshape the contraint vector
    g_ = reshape(g,n,N-1)

    # Pull out values
    fVal = dynamics(prob, X, U)
    Xm = traj_points(prob, X, U, fVal)
    fValm = zero(fVal[1])

    for k = 1:N-1
        Um = (U[k] + U[k+1])*0.5
        evaluate!(fValm, prob.model, Xm[k], Um) # dynamics at the midpoint
        g_[:,k] = -X[k+1] + X[k] + dt*(fVal[k] + 4*fValm + fVal[k+1])/6
    end
end

function collocation_constraints!(g, prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}, X, U) where T
    n,m,N = size(prob)
    dt = prob.dt
    fVal = solver.fVal  #[zero(X[1]) for k = 1:N]
    Xm = solver.X_  #[zero(X[1]) for k = 1:N-1]
    g_colloc = reshape(g, n, N-1)

    # Calculate midpoints
    for k = 1:N
        evaluate!(fVal[k], prob.model, X[k], U[k])
    end
    for k = 1:N-1
        Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
    end
    fValm = zero(X[1])
    for k = 1:N-1
        Um = (U[k] + U[k+1])*0.5
        evaluate!(fValm, prob.model, Xm[k], Um)
        g_colloc[:,k] = -X[k+1] + X[k] + dt*(fVal[k] + 4*fValm + fVal[k+1])/6
    end
end

function get_jac_block(jac::AbstractMatrix, k::Int, n::Vector{Int}, m::Vector{Int},
        ns::Vector{Int}, ms::Vector{Int})
    off1 = sum(ns[1:k-1])
    off2 = sum(ms[1:k-1])
    block = view(jac, off1 .+ (1:n[k]), off2 .+ (1:m[k]))
end

function get_jac_block(jac::AbstractVector, k::Int, n::Vector{Int}, m::Vector{Int},
        ns::Vector{Int}, ms::Vector{Int})
    n_blk = n[k]*m[k]
    off = sum(n[1:k-1] .* m[1:k-1])
    block = reshape(view(jac, off .+ (1:n_blk)), n[k], m[k])
end

# Calculate jacobian
function calc_block!(vals::PartedMatrix, F1,F2,Fm,dt)
    n,m = size(F1.xu)
    In = Diagonal(I, n)
    Im = Diagonal(I, m)
    vals.x1 .= dt/6*(F1.xx + 4Fm.xx*( dt/8*F1.xx + In/2)) + In
    vals.u1 .= dt/6*(F1.xu + 4Fm.xx*( dt/8*F1.xu) + 4Fm.xu*(Im/2))
    vals.x2 .= dt/6*(F2.xx + 4Fm.xx*(-dt/8*F2.xx + In/2)) - In
    vals.u2 .= dt/6*(F2.xu + 4Fm.xx*(-dt/8*F2.xu) + 4Fm.xu*(Im/2))
    return nothing
end

function collocation_constraint_jacobian!(jac, prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}, X, U) where T
    n,m,N = size(prob)
    dt = prob.dt

    # Compute dynamics jacobians
    F = solver.∇F
    for k = 1:N
        jacobian!(F[k], prob.model, X[k], U[k])
    end

    # Calculate midpoints
    fVal = solver.fVal  # [zeros(n) for k = 1:N]
    Xm = solver.X_  # [zeros(n) for k = 1:N-1]
    for k = 1:N
        evaluate!(fVal[k], prob.model, X[k], U[k])
    end
    for k = 1:N-1
        Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
    end


    # Collocation jacobians
    Fm = PartedMatrix(prob.model)
    n_blk = 2(n+m)n
    off = 0
    In = Matrix(I,n,n)
    Im = Matrix(I,m,m)
    part = create_partition2((n,),(n,m,n,m), Val((:x1,:u1,:x2,:u2)))
    for k = 1:N-1
        block = PartedArray(reshape(view(jac, off .+ (1:n_blk)), n, 2(n+m)), part)
        Um = (U[k] + U[k+1])/2
        jacobian!(Fm, prob.model, Xm[k], Um)
        calc_block!(block, F[k], F[k+1], Fm, dt)
        off += n_blk
    end
end

function collocation_constraint_jacobian!(jac, prob::Problem, X::AbstractVectorTrajectory, U::AbstractVectorTrajectory) where T
    n,m,N = size(prob)
    dt = prob.dt
    fVal = dynamics(prob, X, U)
    Xm = traj_points(prob, X, U, fVal)
    ∇F = dynamics_jacobian(prob, X, U)
    ∇Fm = zero(∇F[1])

    In = Matrix(I,n,n)
    Im = Matrix(I,m,m)
    part = create_partition2((n,),(n,m,n,m), Val((:x1,:u1,:x2,:u2)))

    function calc_block!(vals::PartedMatrix, F1,F2,Fm,dt)
        vals.x1 .= dt/6*(F1.xx + 4Fm.xx*( dt/8*F1.xx + In/2)) + In
        vals.u1 .= dt/6*(F1.xu + 4Fm.xx*( dt/8*F1.xu) + 4Fm.xu*(Im/2))
        vals.x2 .= dt/6*(F2.xx + 4Fm.xx*(-dt/8*F2.xx + In/2)) - In
        vals.u2 .= dt/6*(F2.xu + 4Fm.xx*(-dt/8*F2.xu) + 4Fm.xu*(Im/2))
        return nothing
    end

    b1 = ones(Int,N-1)*n
    b2 = ones(Int,N-1)*2(n+m)
    ns = ones(Int,N-1)*n
    ms = ones(Int,N-1)*(n+m)
    for k = 1:N-1

        xm,um = Xm[k], 0.5*(U[k] + U[k+1])
        F1,F2 = ∇F[k], ∇F[k+1]
        jacobian!(∇Fm, prob.model, xm, um)

        block = PartedArray(get_jac_block(jac, k, b1, b2, ns, ms), part)
        calc_block!(block, F1,F2,∇Fm,dt)
    end
end

function collocation_constraint_jacobian_sparsity!(jac::AbstractMatrix, prob::Problem)
    n,m,N = size(prob)
    n_blk = 2(n+m)n

    blk = 1:n_blk
    b1 = ones(Int,N-1)*n
    b2 = ones(Int,N-1)*2(n+m)
    ns = ones(Int,N-1)*n
    ms = ones(Int,N-1)*(n+m)
    off = 0
    for k = 1:N-1
        block = get_jac_block(jac, k, b1, b2, ns, ms)
        block .= reshape(off .+ blk, b1[k], b2[k])
        off += n_blk
    end
end

""" $(SIGNATURES)
Get the row and column lists of a sparse matrix, with ordered elements
"""
function get_rc(A::SparseMatrixCSC)
    row,col,inds = findnz(A)
    v = sortperm(inds)
    row[v],col[v]
end

get_N(prob::Problem, solver::DIRCOLSolver) = get_N(prob.N, solver.opts.method)
