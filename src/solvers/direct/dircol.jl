
""" $(SIGNATURES)
Get the row and column lists of a sparse matrix, with ordered elements
"""
function get_rc(A::SparseMatrixCSC)
    row,col,inds = findnz(A)
    v = sortperm(inds)
    row[v],col[v]
end

function convertInf!(A::VecOrMat{Float64},infbnd=1.1e20)
    infs = isinf.(A)
    A[infs] = sign.(A[infs])*infbnd
    return nothing
end

"""Number of collocation constraints"""
num_colloc(prob::Problem)::Int = (prob.N-1)*prob.model.n


#######################
#   COST FUNCTIONS    #
#######################

cost(prob::Problem, solver::DIRCOLSolver) = cost(prob, solver.Z)

cost(prob::Problem, Z::Primals) = cost(prob.obj, Z.X, Z.U)


##############################
#   COST FUNCTION GRADIENT   #
##############################

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


############################
#    DYNAMICS FUNCTIONS    #
############################

function traj_points!(prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}, X, U) where T
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

function TrajectoryOptimization.dynamics!(prob::Problem{T,Continuous}, solver::DirectSolver, X, U) where T<:AbstractFloat
    for k = 1:prob.N
        evaluate!(solver.fVal[k], prob.model, Z.X[k], Z.U[k])
    end
end


#####################
#    CONSTRAINTS    #
#####################

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

function constraint_jacobians(prob::Problem, solver::DirectSolver, X::AbstractVectorTrajectory, U::AbstractVectorTrajectory)
    n,m,N = size(prob)
    for k = 1:N
        if k == prob.N
            jacobian!(solver.∇C[k], prob.constraints[k], X[k])
        else
            jacobian!(solver.∇C[k], prob.constraints[k], X[k], U[k])
        end
    end
end

function partition_constraint_jacobian(jac::AbstractMatrix, prob::Problem)
    n,m,N = size(prob)
    p_colloc = (N-1)*n
    jac_colloc = view(jac, 1:p_colloc, :)
    jac_custom = view(jac, p_colloc+1:size(jac,1), :)
    return jac_colloc, jac_custom
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
    p_colloc = num_colloc(prob)
    nG_colloc = p_colloc*2(n+m)
    jac_colloc, jac_custom = partition_constraint_jacobian(jac, prob)
    collocation_constraint_jacobian_sparsity!(jac_colloc, prob)

    p = num_constraints(prob)
    off1 = 0
    off2 = 0
    off = nG_colloc
    for k = 1:N
        k == N ? b2 = n : b2 = n+m
        n_blk = p[k]*b2
        block = view(jac_custom, off1 .+ (1:p[k]), (off2 .+ (1:b2)))
        block .= reshape(off .+ (1:n_blk), p[k], b2)
        off1 += p[k]
        off2 += b2
        off += n_blk
    end
end


#################################
#    COLLOCATION CONSTRAINTS    #
#################################

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

function collocation_constraint_jacobian_sparsity!(jac::AbstractMatrix, prob::Problem)
    n,m,N = size(prob)
    n_blk = 2(n+m)n

    blk = 1:n_blk
    off1 = 0
    off2 = 0
    for k = 1:N-1
        block = view(jac, off1 .+ (1:n), off2 .+ (1:2(n+m)))
        block .= reshape((k-1)*n_blk .+ blk, n, 2(n+m))
        off1 += n
        off2 += n+m
    end
end
