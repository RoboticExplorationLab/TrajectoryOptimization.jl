function max_violation_dynamics(prob::Problem{T,Discrete})::T where T <: AbstractFloat
    n = prob.model.n; m = prob.model.m
    max_viol = norm(prob.X[1] - prob.x0,Inf)
    X̄ = zeros(n)
    for k = 1:prob.N-1
        X̄ .= 0
        evaluate!(X̄, prob.model, prob.X[k], prob.U[k][1:m], get_dt(prob,prob.U[k]))
        max_viol = max(max_viol,norm(X̄ - prob.X[k+1],Inf))
    end
    return max_viol
end

# assume dircol solve for continuous problems and defaults to implicit rk3 integration
function max_violation_dynamics(prob::Problem{T,Continuous})::T where T <:AbstractFloat
    n,m,N = size(prob)
    X̄ = zeros(prob.model.n)

    fVal = [zeros(prob.model.n) for k = 1:N]
    fValm = [zeros(prob.model.n) for k = 1:N]
    Xm = [zeros(prob.model.n) for k = 1:N]
    Um = [zeros(prob.model.m) for k = 1:N-1]
    dt = get_dt_traj(prob,prob.U)

    # Calculate midpoints
    for k = 1:N
        evaluate!(fVal[k], prob.model, prob.X[k], prob.U[k])
    end
    for k = 1:N-1
        Xm[k] = (prob.X[k] + prob.X[k+1])/2 + dt[k]/8*(fVal[k] - fVal[k+1])
        Um[k] = (prob.U[k][1:m] + prob.U[k+1][1:m])*0.5
        evaluate!(fValm[k], prob.model, Xm[k], Um[k])
    end

    max_viol = norm(prob.X[1] - prob.x0,Inf)

    for k = 1:N-1
        mv = norm(-prob.X[k+1] + prob.X[k] + dt[k]*(fVal[k] + 4*fValm[k] + fVal[k+1])/6,Inf)
        max_viol = max(max_viol,mv)
    end

    return max_viol
end

function max_violation_direct(prob::Problem)
    max(max_violation(prob),max_violation_dynamics(prob))
end

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

"Generate state midpoint according to quadrature rule"
function gen_xm_cubic(prob::Problem)
    ẋ = zeros(prob.model.n); ẏ = zeros(prob.model.n)

    function xm(y,x,v,u,h)
        prob.model.f(ẋ,x,u)
        prob.model.f(ẏ,y,v)

        0.5*(y+x) + h/8*(ẋ - ẏ)
    end
end

function gen_stage_cost(prob::Problem)
    xm = gen_xm_cubic(prob)
    obj = prob.obj

    function cost(X,U,H)
        N = length(X)
        J = 0.0
        for k = 1:N-1
            Xm = xm(X[k+1],X[k],U[k+1],U[k],H[k])
            Um = 0.5*(U[k] + U[k+1])
            J += H[k]/6*(stage_cost(obj[k],X[k],U[k]) + 4*stage_cost(obj[k],Xm,Um) + stage_cost(obj[k],X[k+1],U[k+1]))
        end
        J += stage_cost(obj[N],X[N])
        return J
    end
end

function gen_stage_cost_gradient(prob::Problem)
    n = prob.model.n; m = prob.model.m; N = prob.N
    function fc(z)
        ż = zeros(eltype(z),n)
        prob.model.f(ż,z[1:n],z[n .+ (1:m)])
        return ż
    end

    function fc(x,u)
        ẋ = zero(x)
        prob.model.f(ẋ,x,u)
        return ẋ
    end

    ∇fc(z) = ForwardDiff.jacobian(fc,z)
    ∇fc(x,u) = ∇fc([x;u])
    dfcdx(x,u) = ∇fc(x,u)[:,1:n]
    dfcdu(x,u) = ∇fc(x,u)[:,n .+ (1:m)]

    xm(y,x,v,u,h) = 0.5*(y + x) + h/8*(fc(x,u) - fc(y,v))
    dxmdy(y,x,v,u,h) = 0.5*I - h/8*dfcdx(y,v)
    dxmdx(y,x,v,u,h) = 0.5*I + h/8*dfcdx(x,u)
    dxmdv(y,x,v,u,h) = -h/8*dfcdu(y,v)
    dxmdu(y,x,v,u,h) = h/8*dfcdu(x,u)



    dℓdx(obj,x,u) = obj.Q*x + obj.q + obj.H'*u
    dℓdu(obj,x,u) = obj.R*u + obj.r + obj.H*x

    dgdx(obj,y,x,v,u,h) = h/6*(dℓdx(obj,x,u) + 4*dxmdx(y,x,v,u,h)'*dℓdx(obj,xm(y,x,v,u,h),0.5*(u+v)))
    dgdy(obj,y,x,v,u,h) = h/6*(4.0*dxmdy(y,x,v,u,h)'*dℓdx(obj,xm(y,x,v,u,h),0.5*(u+v))+ dℓdx(obj,y,v))
    dgdu(obj,y,x,v,u,h) = h/6*(dℓdu(obj,x,u) + 4*(dxmdu(y,x,v,u,h)'*dℓdx(obj,xm(y,x,v,u,h),0.5*(u+v)) + 0.5*dℓdu(obj,xm(y,x,v,u,h),0.5*(u+v))))
    dgdv(obj,y,x,v,u,h) = h/6*(4*(dxmdv(y,x,v,u,h)'*dℓdx(obj,xm(y,x,v,u,h),0.5*(u+v)) + 0.5*dℓdu(obj,xm(y,x,v,u,h),0.5*(u+v))) + dℓdu(obj,y,v))


    nn = 2*(n+m)
    _tmp_ = zeros(n)

    function _cost_grad!(∇g,X,U,H)
        shift = 0
        ∇g .= 0 # set all to zero, for additive inplace operations
        for k = 1:N-1
            obj = prob.obj[k]
            x = X[k]; y = X[k+1]; u = U[k]; v = U[k+1]; h = H[k]

            _xm = xm(y,x,v,u,h)
            _um = 0.5*(u+v)
            _dℓdx = dℓdx(obj,x,u)
            _dℓdu = dℓdu(obj,x,u)
            _dℓdy = dℓdx(obj,y,v)
            _dℓdv = dℓdu(obj,y,v)
            _dℓmdx = dℓdx(obj,_xm,_um)
            _dℓmdu = dℓdu(obj,_xm,_um)
            _dxmdx = dxmdx(y,x,v,u,h)
            _dxmdy = dxmdy(y,x,v,u,h)
            _dxmdu = dxmdu(y,x,v,u,h)
            _dxmdv = dxmdv(y,x,v,u,h)

            _∇g = view(∇g,shift .+ (1:nn))
            _∇g[1:n] += h/6*(_dℓdx + 4*_dxmdx'*_dℓmdx)
            _∇g[n .+ (1:m)] += h/6*(_dℓdu + 4*(_dxmdu'*_dℓmdx + 0.5*_dℓmdu))
            _∇g[(n+m) .+ (1:n)] += h/6*(4.0*_dxmdy'*_dℓmdx + dℓdx(obj,y,v))
            _∇g[(2*n+m) .+ (1:m)] += h/6*(4*(_dxmdv'*_dℓmdx + 0.5*_dℓmdu) + dℓdu(obj,y,v))
            shift += (n+m)
        end

        gradient!(_tmp_, prob.obj[N], X[N])
        ∇g[(N-1)*(n+m) .+ (1:n)] += _tmp_
    end
end

cost(prob::Problem, solver::DIRCOLSolver) = cost(prob, solver.Z)
cost(prob::Problem, Z::Primals) = cost(prob.obj, Z.X, Z.U, get_dt_traj(prob))


##############################
#   COST FUNCTION GRADIENT   #
##############################

function cost_gradient!(grad_f, prob::Problem, X::AbstractVectorTrajectory, U::AbstractVectorTrajectory, H::Vector)
    n,m,N = size(prob)
    grad = reshape(grad_f, n+m, N)
    part = (x=1:n, u=n .+ (1:m))
    dt = get_dt_traj(prob,U)
    for k = 1:N-1
        grad_k = PartedVector(view(grad,:,k), part)
        gradient!(grad_k, prob.obj[k], X[k], U[k])
        grad_k .*= dt[k]
    end
    grad_k = PartedVector(view(grad,1:n,N), part)
    gradient!(grad_k, prob.obj[N], X[N])
    return nothing
end


############################
#    DYNAMICS FUNCTIONS    #
############################

# function traj_points!(prob::Problem, solver::DIRCOLSolver{T,HermiteSimpson}, X, U) where T
#     n,m,N = size(prob)
#     dt = prob.dt
#     Xm = solver.X_
#     fVal = solver.fVal
#     X,U = Z.X, Z.U
#     for k = 1:N-1
#         Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
#     end
#     return Xm
# end
#
# function TrajectoryOptimization.dynamics!(prob::Problem{T,Continuous}, solver::DirectSolver, X, U) where T<:AbstractFloat
#     for k = 1:prob.N
#         evaluate!(solver.fVal[k], prob.model, X[k], U[k])
#     end
# end


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

# function constraint_jacobians(prob::Problem, solver::DirectSolver, X::AbstractVectorTrajectory, U::AbstractVectorTrajectory)
#     n,m,N = size(prob)
#     for k = 1:N
#         if k == prob.N
#             jacobian!(solver.∇C[k], prob.constraints[k], X[k])
#         else
#             jacobian!(solver.∇C[k], prob.constraints[k], X[k], U[k])
#         end
#     end
# end

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
    fValm = copy(fVal[1])
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

# generate DIRCOL functions
function gen_dircol_functions(prob::Problem{T}, solver::DIRCOLSolver) where T
    n,m,N = size(prob)
    NN = N*(n+m)
    p_colloc = num_colloc(prob)
    p = num_constraints(prob)
    P = p_colloc + sum(p)
    dt = prob.dt

    part_f = create_partition2(prob.model)
    part_z = create_partition(n,m,N,N)
    pcum = cumsum(p)

    jac_structure = spzeros(P, NN)
    constraint_jacobian_sparsity!(jac_structure, prob)
    r,c = get_rc(jac_structure)

    function eval_f(Z)
        X,U = unpack(Z,part_z)
        cost(prob.obj, X, U, get_dt_traj(prob))
    end

    function eval_grad_f(Z, grad_f)
        X,U = unpack(Z, part_z)
        cost_gradient!(grad_f, prob, X, U, get_dt_traj(prob))
    end

    function eval_g(Z, g)
        X,U = unpack(Z,part_z)
        g_colloc = view(g,1:p_colloc)
        g_custom = view(g,p_colloc+1:length(g))

        collocation_constraints!(g_colloc, prob, solver, X, U)
        update_constraints!(g_custom, prob, solver, X, U)
    end


    function eval_jac_g(Z, mode, rows, cols, vals)
        if mode == :Structure
            copyto!(rows,r)
            copyto!(cols,c)
        else
            X,U = unpack(Z, part_z)

            nG_colloc = p_colloc * 2(n+m)
            jac_colloc = view(vals, 1:nG_colloc)
            collocation_constraint_jacobian!(jac_colloc, prob, solver, X, U)

            # General constraint jacobians
            jac_custom = view(vals, nG_colloc+1:length(vals))
            constraint_jacobian!(jac_custom, prob, solver, X, U)
        end

        return nothing
    end
    return eval_f, eval_g, eval_grad_f, eval_jac_g
end

function remove_bounds!(prob::Problem)
    n,m,N = size(prob)
    bounds = [BoundConstraint(n,m) for k = 1:prob.N]

    # Initial Time step
    if :bound ∈ labels(prob.constraints[1])
        bnd_init = remove_bounds!(prob.constraints[1])[1]
    else
        bnd_init = bounds[1]
    end
    bounds[1] = BoundConstraint(n,m, x_min=prob.x0, u_min=bnd_init.u_min,
                                     x_max=prob.x0, u_max=bnd_init.u_max)

    # All time steps
    for k = 2:prob.N
        bnd = remove_bounds!(prob.constraints[k])
        if !isempty(bnd)
            bounds[k] = bnd[1]::BoundConstraint
        end
    end
if :goal ∈ labels(prob.constraints[N])
        goal = pop!(prob.constraints[N])
        xf = zeros(n)
        evaluate!(xf, goal, zero(xf))
        term_bound = BoundConstraint(n,m, x_min=-xf, u_min=bounds[N-1].u_min,
                                          x_max=-xf, u_max=bounds[N-1].u_max)
        bounds[N] = term_bound::BoundConstraint
    end
    # Terminal time step
    #TODO handle control at Nth U differently

    if :goal ∈ labels(prob.constraints[N])
        goal = pop!(prob.constraints[N])
        xf = zeros(n)
        evaluate!(xf, goal, zero(xf))
        term_bound = BoundConstraint(n,m, x_min=-xf, u_min=bounds[N-1].u_min,
                                          x_max=-xf, u_max=bounds[N-1].u_max)
        bounds[N] = term_bound::BoundConstraint
    end
    return bounds
end

function get_bounds(prob::Problem, bounds::Vector{<:BoundConstraint})
    n,m,N = size(prob)
    p_colloc = num_colloc(prob)
    Z = Primals(prob, true)

    Z.equal ? uN = N : uN = N-1
    x_U = [zeros(n) for k = 1:N]
    x_L = [zeros(n) for k = 1:N]
    u_U = [zeros(m) for k = 1:uN]
    u_L = [zeros(m) for k = 1:uN]
    for k = 1:uN
        x_U[k] = bounds[k].x_max
        x_L[k] = bounds[k].x_min
        u_U[k] = bounds[k].u_max
        u_L[k] = bounds[k].u_min
    end
    #TODO handle control at Nth U differently
    if Z.equal
        u_U[N] = bounds[N-1].u_max
        u_L[N] = bounds[N-1].u_min
    end
    if !Z.equal
        x_U = bounds[N].x_max
        x_L = bounds[N].x_min
    end
    z_U = Primals(x_U,u_U)
    z_L = Primals(x_L,u_L)

    # Constraints
    p = num_constraints(prob)
    g_U = [PartedVector(prob.constraints[k]) for k = 1:N-1]
    g_L = [PartedVector(prob.constraints[k]) for k = 1:N-1]
    push!(g_U, PartedVector(prob.constraints[N], :terminal))
    push!(g_L, PartedVector(prob.constraints[N], :terminal))
    for k = 1:N
        if p[k] > 0
            g_L[k].inequality .= -Inf
        end
    end
    g_U = vcat(zeros(p_colloc), g_U...)
    g_L = vcat(zeros(p_colloc), g_L...)

    convertInf!(z_U.Z)
    convertInf!(z_L.Z)
    convertInf!(g_U)
    convertInf!(g_L)
    return z_U.Z, z_L.Z, g_U, g_L
end
