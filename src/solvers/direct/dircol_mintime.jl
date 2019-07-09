"Add row and column indices to existing lists"
function add_rows_cols!(row,col,_r,_c)
    for cc in _c
        for rr in _r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
end


#######################
#   COST FUNCTIONS    #
#######################

function gen_stage_cost_min_time(prob::Problem, R_min_time::T) where T
    xm = gen_xm_cubic(prob)
    obj = prob.obj

    function cost(X,U,H)
        N = length(X)
        J = 0.0
        for k = 1:N-1
            Xm = xm(X[k+1],X[k],U[k+1],U[k],H[k])
            Um = 0.5*(U[k] + U[k+1])
            J += H[k]/6*(stage_cost(obj[k],X[k],U[k]) + 4*stage_cost(obj[k],Xm,Um) + stage_cost(obj[k],X[k+1],U[k+1])) + R_min_time*H[k]
        end
        J += stage_cost(obj[N],X[N])
        return J
    end
end

function gen_stage_cost_gradient_min_time(prob::Problem,R_min_time::T) where T
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
    dxmdh(y,x,v,u,h) = 1/8*(fc(x,u) - fc(y,v))

    dℓdx(obj,x,u) = obj.Q*x + obj.q + obj.H'*u
    dℓdu(obj,x,u) = obj.R*u + obj.r + obj.H*x

    dgdx(obj,y,x,v,u,h) = h/6*(dℓdx(obj,x,u) + 4*dxmdx(y,x,v,u,h)'*dℓdx(obj,xm(y,x,v,u,h),0.5*(u+v)))
    dgdy(obj,y,x,v,u,h) = h/6*(4.0*dxmdy(y,x,v,u,h)'*dℓdx(obj,xm(y,x,v,u,h),0.5*(u+v))+ dℓdx(obj,y,v))
    dgdu(obj,y,x,v,u,h) = h/6*(dℓdu(obj,x,u) + 4*(dxmdu(y,x,v,u,h)'*dℓdx(obj,xm(y,x,v,u,h),0.5*(u+v)) + dℓdu(obj,xm(y,x,v,u,h),0.5*(u+v))))
    dgdv(obj,y,x,v,u,h) = h/6*(4*(dxmdu(y,x,v,u,h)'*dℓdx(obj,xm(y,x,v,u,h),0.5*(u+v)) + dℓdu(obj,xm(y,x,v,u,h),0.5*(u+v))) + dℓdu(obj,y,v))
    dgdh(obj,y,x,v,u,h) = 1/6*(stage_cost(obj,x,u) + 4*stage_cost(obj,xm(y,x,v,u,h),0.5*(u+v)) + stage_cost(obj,y,v)) + 4*h/6*dxmdh(y,x,v,u,h)'*dℓdx(xm(y,x,v,u,h),0.5*(u+v)) + R_min_time

    nn = 2*(n+m) + 1
    _tmp_ = zeros(n)

    function _cost_grad!(∇g,X,U,H)
        shift = 0
        ∇g .= 0 # set all to zero, for additive inplace operations
        for k = 1:N-1
            obj = prob.obj[k]
            x = X[k]; y = X[k+1]; u = U[k]; v = U[k+1]; h = H[k]
            ∇g[shift .+ (1:nn)][1:n] += dgdx(obj,y,x,v,u,h)
            ∇g[shift .+ (1:nn)][n .+ (1:m)] += dgdu(obj,y,x,v,u,h)
            ∇g[shift .+ (1:nn)][n+m+1] += dgdh(obj,y,x,v,u,h)
            ∇g[shift .+ (1:nn)][(n+m+1) .+ (1:n)] += dgdy(obj,y,x,v,u,h)
            ∇g[shift .+ (1:nn)][(2*n+m+1) .+ (1:m)] += dgdv(obj,y,x,v,u,h)

            shift += (n+m+1)
        end

        gradient!(_tmp_, prob.obj[N], X[N])
        ∇g[(N-1)*(n+m+1) .+ (1:n)] += _tmp_
    end
end

#####################
#    CONSTRAINTS    #
#####################

function update_constraints!(g, prob::Problem, solver::DIRCOLSolverMT, X, U) where T
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

function custom_constraint_jacobian_sparsityMT!(prob::Problem,r_shift=0)
    n,m,N = size(prob)
    row = []
    col = []

    c_shift = 0

    p = num_constraints(prob)

    for k = 1:N-1
        r_idx = r_shift .+ (1:p[k])
        c_idx = c_shift .+ (1:(n+m))
        add_rows_cols!(row,col,r_idx,c_idx)
        r_shift += p[k]
        c_shift += (n+m+1)
    end

    k = N
    r_idx = r_shift .+ (1:p[N])
    c_idx = c_shift .+ (1:n)
    add_rows_cols!(row,col,r_idx,c_idx)
    r_shift += p[N]
    c_shift += (n+m)

    return collect(zip(row,col))
end

#################################
#    COLLOCATION CONSTRAINTS    #
#################################

function collocation_constraints!(g, prob::Problem, solver::DIRCOLSolverMT{T,HermiteSimpson}, X, U) where T
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

function collocation_constraint_jacobian!(jac, prob::Problem, solver::DIRCOLSolverMT{T,HermiteSimpson}, X, U) where T
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

function collocation_constraint_jacobian_sparsityMT!(prob::Problem, r_shift=0)
    n,m,N = size(prob)
    row = []
    col = []

    c_shift = 0
    for k = 1:N-1
        r_idx = r_shift .+ (1:n)
        c_idx = c_shift .+ [(1:(n+m))...,((1+n+m) .+ (1:n+m))...]
        add_rows_cols!(row,col,r_idx,c_idx)
        r_shift += n
        c_shift += (n+m+1)
    end

    return collect(zip(row,col))
end

function h_eq_constraint_jacobian!(jac,prob,solver,H)
    shift = 0
    ∇c = [1.0;-1.0]
    for k = 1:N-2
        jac[shift .+ (1:2)] = ∇c
        shift += 2
    end
end

function h_eq_constraint_sparsityMT!(prob,r_shift=0)
    n,m,N = size(prob)
    row = []
    col = []

    c_shift = n+m+1
    for k = 1:N-2
        r_idx = r_shift
        c_idx = c_shift .+ [0, (n+m+1)]
        add_rows_cols!(row,col,r_idx,c_idx)
        r_shift += 1
        c_shift += (n+m+1)
    end

    return collect(zip(row,col))
end

function get_boundsMT(prob::Problem, bounds::Vector{<:BoundConstraint},h_max,h_min)
    n,m,N = size(prob)
    p_colloc = num_colloc(prob)
    Z = PrimalsMT(prob, true)

    N = length(Z.X)
    uN = length(Z.U)
    hN = length(Z.H)

    Z.equal ? uN = N : uN = N-1
    x_U = [zeros(n) for k = 1:N]
    x_L = [zeros(n) for k = 1:N]
    u_U = [zeros(m) for k = 1:uN]
    u_L = [zeros(m) for k = 1:uN]
    h_U = [zeros(1) for k = 1:hN]
    h_L = [zeros(1) for k = 1:hN]

    for k = 1:uN
        x_U[k] = bounds[k].x_max
        x_L[k] = bounds[k].x_min
        u_U[k] = bounds[k].u_max
        u_L[k] = bounds[k].u_min
        if k <= hN
            h_U[k] = h_max
            h_L[k] = h_min
        end
    end
    if !Z.equal
        x_U = bounds[N].x_max
        x_L = bounds[N].x_min
    end
    z_U = PrimalsMT(x_U,u_U,h_U)
    z_L = PrimalsMT(x_L,u_L,h_L)

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
    g_U = vcat(zeros(p_colloc), g_U..., zeros(N-2))
    g_L = vcat(zeros(p_colloc), g_L..., zeros(N-2))

    convertInf!(z_U.Z)
    convertInf!(z_L.Z)
    convertInf!(g_U)
    convertInf!(g_L)
    return z_U.Z, z_L.Z, g_U, g_L
end
