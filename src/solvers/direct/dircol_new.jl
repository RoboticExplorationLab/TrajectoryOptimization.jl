cost(prob::Problem, solver::DIRCOLSolver) = cost(prob, solver.Z)

cost(prob::Problem, Z::Primals) = cost(prob.obj, Z.X, Z.U)

"""Number of collocation constraints"""
num_colloc(prob::Problem)::Int = (prob.N-1)*prob.model.n

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

function traj_points!(prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}, Z=solver.Z) where T
    n,m,N = size(prob)
    dt = prob.dt
    Xm = solver.X_
    fVal = solver.fVal
    X,U = Z.X, Z.U
    for k = 1:N-1
        Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
    end
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

function calculate_jacobians!(prob::Problem, solver::DirectSolver, Z=solver.Z)
    for k = 1:prob.N
        jacobian!(solver.∇F[k], prob.model,          Z.X[k], Z.U[k])
        if k == prob.N
            jacobian!(solver.∇C[k], prob.constraints[k], Z.X[k])
        else
            jacobian!(solver.∇C[k], prob.constraints[k], Z.X[k], Z.U[k])
        end
    end
end


function update_constraints!(g, prob::Problem, Z)
    n,m,N = size(prob)
    p,pN = num_stage_constraints(prob), num_terminal_constraints(prob)
    P = (N-1)*p
    X,U = Z.X, Z.U

    g_stage = reshape(view(g,1:P), p, N-1)
    g_term = view(g,P+1:length(g))

    for k = 1:N-1
        evaluate!(g_stage[:,k], prob.constraints, X[k], U[k])
    end
    evaluate!(g_term, prob.constraints, X[N])
end

function update_constraints!(g, prob::Problem, solver::DIRCOLSolver, Z::Primals=solver.Z) where T
    n,m,N = size(prob)
    p_colloc = n*(N-1)
    g_colloc = view(g,1:p_colloc)
    collocation_constraints!(g_colloc, prob, solver)

    X,U = solver.Z.X, solver.Z.U
    g_custom = view(g, p_colloc + 1:length(g))
    p = num_constraints(prob.constraints)
    offset = 0
    for k = 1:N
        c = PartedVector(view(g_custom, offset .+ (1:p[k])), solver.C[k].parts)
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

function constraint_jacobian!(jac, prob::Problem, solver::DirectSolver, Z::Primals=solver.Z)
    n,m,N = size(prob)
    p_colloc = num_colloc(prob)
    jac_colloc, jac_custom = partition_constraint_jacobian(jac, prob)
    collocation_constraint_jacobian!(jac_colloc, prob, solver, Z)
    p = num_constraints(prob)

    off1 = p_colloc
    off2 = 0
    for k = 1:N
        if k == N
            block = get_jac_block(jac_custom, k, p[k], n)
        else
            block = get_jac_block(jac_custom, k, p[k], n+m)
        end
        block .= solver.∇C[k]
        off1 += p[k]
        off2 += n+m
    end
end

function constraint_jacobian_sparsity!(jac::AbstractMatrix, prob::Problem)
    n,m,N = size(prob)
    p_colloc = n*(N-1)
    jac_colloc, jac_custom = partition_constraint_jacobian(jac, prob)
    collocation_constraint_jacobian_sparsity!(jac_colloc, prob)

    p = num_constraints(prob)
    off = p_colloc*2(n+m)
    for k = 1:N
        k == N ? b2 = n : b2 = n+m
        n_blk = p[k]*b2
        block = get_jac_block(jac_custom, k, p[k], b2)
        block .= reshape(off .+ (1:n_blk), p[k], b2)
        off += n_blk
    end
end

function collocation_constraints!(g, prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}, Z::Primals=solver.Z) where T n,m,N = size(prob)
    @assert isodd(N)
    dt = prob.dt

    # Reshape the contraint vector
    g_ = reshape(g,n,N-1)

    # Pull out values
    fVal = solver.fVal
    X = Z.X
    U = Z.U
    Xm = solver.X_
    fValm = zero(fVal[1])

    for k = 1:N-1
        Um = (U[k] + U[k+1])*0.5
        evaluate!(fValm, prob.model, Xm[k], Um) # dynamics at the midpoint
        g_[:,k] = -X[k+1] + X[k] + dt*(fVal[k] + 4*fValm + fVal[k+1])/6
    end
end

function get_jac_block(jac::AbstractMatrix, k::Int, n::Int, m::Int, off_n=n, off_m=m)
    off1 = (k-1)*off_n
    off2 = (k-1)*off_m
    block = view(jac, off1 .+ (1:n), off2 .+ (1:m))
end

function get_jac_block(jac::AbstractVector, k::Int, n::Int, m::Int, off_n=n, off_m=m)
    n_blk = n*m
    block = reshape(view(jac, (k-1)*n_blk .+ (1:n_blk)), n, m)
end

function collocation_constraint_jacobian!(jac, prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}, Z=solver.Z) where T
    n,m,N = size(prob)
    dt = prob.dt
    X, U = Z.X, Z.U
    Xm = solver.X_
    ∇F = solver.∇F
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

    b1 = n
    b2 = 2(n+m)
    off_n = n
    off_m = n+m
    for k = 1:N-1

        xm,um = Xm[k], 0.5*(U[k] + U[k+1])
        F1,F2 = ∇F[k], ∇F[k+1]
        jacobian!(∇Fm, prob.model, xm, um)

        block = PartedArray(get_jac_block(jac, k, b1, b2, off_n, off_m), part)
        calc_block!(block, F1,F2,∇Fm,dt)
    end
end

function collocation_constraint_jacobian_sparsity!(jac::AbstractMatrix, prob::Problem)
    n,m,N = size(prob)
    n_blk = 2(n+m)n

    blk = 1:n_blk
    b1, b2 = n, 2(n+m)
    off_n, off_m = n, n+m
    off = 0
    for k = 1:N-1
        block = get_jac_block(jac, k, b1, b2, off_n, off_m)
        block .= reshape(off .+ blk, b1, b2)
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
