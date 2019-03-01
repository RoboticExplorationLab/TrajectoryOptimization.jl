import Base.getindex
# using Snopt

struct PrimalVars{V,T}
    Z::V
    X::SubArray{T}
    U::SubArray{T}
end

function PrimalVars(Z::AbstractVector,ind_x::Matrix{Int},ind_u::Matrix{Int})
    n,N = size(ind_x)
    m = size(ind_u,1)
    Nz = N*n + (N-1)*m
    if length(Z) > Nz
        Z = Z[1:Nz]
    end
    X = view(Z,ind_x)
    U = view(Z,ind_u)
    PrimalVars(Z,X,U)
end

function PrimalVars(Z::AbstractVector,n::Int,m::Int,N::Int)
    ind_x, ind_u = get_primal_inds(n,m,N)
    PrimalVars(Z,ind_x,ind_u)
end

function PrimalVars(n::Int,m::Int,N::Int)
    Z = zeros(N*n + (N-1)*m)
    PrimalVars(Z,n,m,N)
end

function PrimalVars(X::Matrix,U::Matrix)
    Z = packZ(X,U)
    n,N = size(X)
    m = size(U,1)
    PrimalVars(Z,n,m,N)
end

function PrimalVars(res::SolverIterResults)
    PrimalVars(to_array(res.X), to_array(res.U))
end


get_sizes(Z::PrimalVars) = size(Z.X,1), size(Z.U,1), size(Z.X,2)
Base.length(Z::PrimalVars) = length(Z.Z)

function getindex(Z::PrimalVars,k::Int)
    n,m,N = get_sizes(Z)
    if k < N
        return Z.Z[(1:n+m) .+ (k-1)*(n+m)]
    elseif k == N
        return Z.Z[end-n+1:end]
    else
        BoundsError("Cannot retrieve time step $k, Z only has $N time steps")
    end
end

struct NewtonVars{T}
    V::Vector{T}
    Z::PrimalVars{V,T} where V
    ν::SubArray{T}
    λ::SubArray{T}
    μ::SubArray{T}
    a::Vector{Bool}  # Active set
    dualinds::UnitRange{Int}
end

function NewtonVars(Z::PrimalVars,ν::T,λ::T,μ::T) where T <: AbstractVector
    Nz = length(Z)
    Nx = length(ν)
    Nh = length(λ)
    Ng = length(μ)
    n,m,N = get_sizes(Z)
    part = create_partition([Nz,Nx,Nh,Ng],(:z,:ν,:λ,:μ))
    V = [Z.Z;ν;λ;μ]
    Z = PrimalVars(view(V,part.z),n,m,N)
    ν = view(V,part.ν)
    λ = view(V,part.λ)
    μ = view(V,part.μ)
    a = ones(Bool,length(V))
    dualinds = Nz .+ (1:Nx+Nh+Ng)
    NewtonVars(V,Z,ν,λ,μ,a,dualinds)
end

primals(V::NewtonVars) = V.Z.Z
duals(V::NewtonVars) = view(V.V,V.dualinds)
duals(V::NewtonVars, a::Vector{Bool}) = view(duals(V),a[V.dualinds])
Base.copy(V::NewtonVars) = NewtonVars(V.Z,V.ν,V.λ,V.μ)
+(V::NewtonVars,A::Vector) = begin V1 = copy(V); V1.V .+= A; return V1 end
Base.length(V::NewtonVars) = length(V.V)

function NewtonVars(solver::Solver, res::SolverIterResults)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    X = to_array(res.X)
    U = to_array(res.U)
    m = size(U,1)
    ν = zeros(n,N)

    for k = 1:N
        ν[:,k] = res.s[k]*0
    end
    ν = vec(ν)

    if res isa ConstrainedVectorResults
        s = zeros(pI,N-1)
        μ = zeros(pI,N-1)
        λ = zeros(pE,N-1)
        ineq = 1:pI
        eq = pI .+ (1:pE)
        for k = 1:N-1
            μ[:,k] = res.λ[k][ineq]
            λ[:,k] = res.λ[k][eq]
        end
        s = vec(s)
        μ = vec(μ)
        λ = vec(λ)
        append!(s, sqrt.(2.0*max.(0,-res.C[N][1:pI_N])))
        append!(μ, res.λ[N][1:pI_N])
        append!(λ, res.λ[N][pI_N .+ (1:pE_N)])
    else
        λ = Float64[]
        μ = Float64[]
    end
    Z = PrimalVars(X,U)
    NewtonVars(Z,ν,λ,μ)
end

function get_primal_inds(n,m,N)
    Nx = N*n
    Nu = (N-1)*m
    Nz = Nx+Nu
    ind_x = zeros(Int,n,N)
    ind_u = zeros(Int,m,N-1)
    ix = 1:n
    iu = n .+ (1:m)
    for k = 1:N-1
        ind_x[:,k] = ix .+ (k-1)*(n+m)
        ind_u[:,k] = iu .+ (k-1)*(n+m)
    end
    ind_x[:,N] = ix .+ (N-1)*(n+m)
    return ind_x, ind_u
end

function get_batch_sizes(solver::Solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
    n,m,N = get_sizes(solver)

    Nx = N*n
    Nu = (N-1)*m
    Nh = (N-1)*pE + pE_N
    Ng = (N-1)*pI + pI_N
    NN = 2Nx + Nu + Nh + Ng
    return Nx,Nu,Nh,Ng,NN
end

function gen_usrfun_newton(solver::Solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
    n,m,N = get_sizes(solver)

    c_function!, c_jacobian!, c_labels, cI!, cE! = generate_constraint_functions(solver.obj)
    obj = solver.obj
    costfun = obj.cost


    Nx,Nu,Nh,Ng,NN = get_batch_sizes(solver)
    Nz = Nx+Nu

    names = (:z,:ν,:λ,:μ)
    ind1 = TrajectoryOptimization.create_partition([Nz,Nx,Nh,Ng],names)
    ind2 = TrajectoryOptimization.create_partition2([Nz,Nx,Nh,Ng],names)

    ind_x, ind_u = get_primal_inds(n,m,N)
    ind_Z = create_partition([Nx,Nu],(:x,:u))
    ind_z = create_partition([n,m],(:x,:u))
    ind_h = create_partition([(N-1)*pE,pE_N],(:k,:N))
    ind_g = create_partition([(N-1)*pI,pI_N],(:k,:N))
    ind_h = (k=ind_h.k, N=ind_h.N, k_mat=reshape(ind_h.k,pE,N-1))
    ind_g = (k=ind_g.k, N=ind_g.N, k_mat=reshape(ind_g.k,pI,N-1))

    ind_v = create_partition([Nz,Nx,Nh,Ng],(:z,:ν,:λ,:μ))

    if solver.obj.cost isa QuadraticCost
        Z0 = PrimalVars(n,m,N)
        ∇²J, = taylor_expansion(solver,Z0)
    else
        error("Not yet implemented for non-Quadratic cost functions")
    end

    # Set up jacobians (fast for this application)
    discretizer = eval(solver.integration)
    f!(xdot,x,u) = TrajectoryOptimization.dynamics(solver.model,xdot,x,u)
    fd! = rk4(f!,solver.dt)
    fd_aug!(xdot,z) = fd!(xdot,view(z,ind_z.x),view(z,ind_z.u))
    Fd!(dz,xdot,z) = ForwardDiff.jacobian!(dz,fd_aug!,xdot,z)

    # Bounds
    z_L,z_U = get_bounds(solver,false)
    x_bnd = [isfinite.(obj.x_max); -isfinite.(obj.x_min)]
    u_bnd = [isfinite.(obj.u_max); -isfinite.(obj.u_min)]
    z_bnd = [u_bnd; x_bnd]
    active_bnd = z_bnd .!= 0
    jac_x_bnd= [Diagonal(isfinite.(obj.x_max));
               -Diagonal(isfinite.(obj.x_min))]
    jac_u_bnd = [Diagonal(isfinite.(obj.u_max));
                -Diagonal(isfinite.(obj.u_min))]
    jac_bnd_k = [spzeros(2m,n) jac_u_bnd;
                 jac_x_bnd spzeros(2n,m)][active_bnd,:]
    # jac_bnd_k = blockdiag(jac_x_bnd,jac_u_bnd)[active_bnd,:]
    jac_bnd = blockdiag([jac_bnd_k for k = 1:N-1]...)
    # jac_bnd = blockdiag(jac_bnd,jac_x_bnd[active_bnd[2m+1:end],:])
    jac_bnd = [jac_bnd spzeros(Ng,n)]


    function mycost(Z::PrimalVars)
        X,U = Z.X, Z.U
        J = 0.0
        for k = 1:N-1
            J += stage_cost(costfun,X[:,k],U[:,k])*solver.dt
        end
        J += stage_cost(costfun,X[:,N])
        return J
    end
    mycost(Z::AbstractVector) = mycost(PrimalVars(Z,ind_x,ind_u))
    mycost(V::NewtonVars) = mycost(V.Z)

    function cE(C,Z)
        Z̄ = PrimalVars(Z,ind_x,ind_u)
        X,U = Z̄.X, Z̄.U

        C_ = view(C,ind_h.k_mat)
        for k = 1:N-1
            cE!(view(C_,1:pE,k),X[:,k],U[:,k])
        end
        CN = view(C,ind_h.N)
        cE!(CN,X[:,N])
        return nothing
    end
    function cE(Z)
        C = zeros(eltype(Z),Nh)
        cE(C,Z)
        return C
    end

    function cI(C,Z)
        Z̄ = PrimalVars(Z,ind_x,ind_u)
        X,U = Z̄.X, Z̄.U

        C_ = view(C,ind_g.k_mat)
        for k = 1:N-1
            cI!(view(C_,1:pI,k),X[:,k],U[:,k])
        end
        CN = view(C,ind_g.N)
        cI!(CN,X[:,N])
        return nothing
    end
    function cI(Z)
        C = zeros(eltype(Z),Ng)
        cI(C,Z)
        return C
    end

    function dynamics(D,Z)
        Z̄ = PrimalVars(Z,ind_x,ind_u)
        X,U = Z̄.X, Z̄.U

        d = reshape(D,n,N)
        d[:,1] = X[:,1] - solver.obj.x0
        for k = 2:N
            solver.fd(view(d,1:n,k),X[:,k-1],U[:,k-1])
            d[:,k] -= X[:,k]
        end
    end
    function dynamics(Z)
        D = zeros(eltype(Z),Nx)
        dynamics(D,Z)
        return D
    end

    grad_J(grad,Z) = gradient!(grad,solver,PrimalVars(Z,ind_x,ind_u))
    grad_J(Z) = gradient(solver,PrimalVars(Z,ind_x,ind_u))

    hess_J(hess,Z) = copyto!(hess,∇²J)
    hess_J(Z) = ∇²J

    function jacob_dynamics(jacob,Z)
        Z̄ = PrimalVars(Z,ind_x,ind_u)
        xdot = zeros(n)
        jacob[1:n,1:n] = Diagonal(I,n)
        for k = 1:N-1
            off1 = k*n
            off2 = (k-1)*(n+m)
            block = view(jacob,off1 .+ (1:n),off2 .+ (1:n+m))
            Fd!(block, xdot, Z̄[k])
            jacob[off1 .+ (1:n),(off2+n+m) .+ (1:n)] = -Diagonal(I,n)
        end
    end
    function jacob_dynamics(Z)
        jacob = spzeros(Nx,Nz)
        jacob_dynamics(jacob,Z)
        return jacob
    end
    # Sparsity structure (1s in all non-zero elements)
    function jacob_dynamics()
        jacob = spzeros(Nx,Nz)
        jacob[1:n,1:n] = Diagonal(I,n)
        for k = 1:N-1
            off1 = k*n
            off2 = (k-1)*(n+m)
            block = view(jacob,off1 .+ (1:n),off2 .+ (1:n+m))
            block .= 1
            jacob[off1 .+ (1:n),(off2+n+m) .+ (1:n)] = Diagonal(I,n)
        end
        return jacob
    end

    function jacob_cE(jacob,Z)
        if obj.use_xf_equality_constraint
            jacob[:,Nz-n+1:Nz] = Diagonal(I,n)
        end
    end
    function jacob_cE(Z)
        jacob = spzeros(Nh,Nz)
        jacob_cE(jacob,Z)
        return jacob
    end
    # Assume only bound constraints right now
    jacob_cI(jacob,Z) = copyto!(jacob,jac_bnd)
    jacob_cI(Z) = jac_bnd

    c = (E=cE, I=cI, d=dynamics)
    jacob_c = (E=jacob_cE, I=jacob_cI, d=jacob_dynamics)

    return mycost, c, grad_J, hess_J, jacob_c
end


"""$(SIGNATURES)
Full 2nd order taylor expansion of a Quadratic Cost with respect to Z = [X;U]
"""
function taylor_expansion(solver::Solver,Z::PrimalVars)
    costfun = solver.obj.cost
    dt = solver.dt
    n,m,N = get_sizes(Z)
    Nx = N*n
    Nu = (N-1)*m
    Nz = Nx + Nu
    hess = spzeros(Nz,Nz)
    grad = spzeros(Nz)

    ind1 = create_partition([n,m],(:x,:u))

    X,U = Z.X, Z.U
    for k = 1:N-1
        off = (k-1)*(n+m)
        Q,R,H,q,r = taylor_expansion(costfun,X[:,k],U[:,k])
        hess[off .+ ind1.x, off .+ ind1.x] = Q
        hess[off .+ ind1.x, off .+ ind1.u] = H'
        hess[off .+ ind1.u, off .+ ind1.x] = H
        hess[off .+ ind1.u, off .+ ind1.u] = R

        grad[off .+ ind1.x] = q
        grad[off .+ ind1.u] = r
    end
    hess .*= dt
    grad .*= dt
    off = (N-1)*(n+m)
    Qf, qf = taylor_expansion(costfun,X[:,N])
    hess[off .+ ind1.x, off .+ ind1.x] = Qf
    grad[off .+ ind1.x] = qf
    return hess, grad
end

"""
$(SIGNATURES)
Full 1st order taylor expansion of a Quadratic Cost with respect to Z = [X;U]
"""
function gradient!(grad::AbstractVector,solver::Solver,Z::PrimalVars)
    costfun = solver.obj.cost
    dt = solver.dt

    n,m,N = get_sizes(Z)
    Nx = N*n
    Nu = (N-1)*m
    Nz = Nx + Nu
    ind1 = create_partition([n,m],(:x,:u))

    X,U = Z.X, Z.U
    for k = 1:N-1
        off = (k-1)*(n+m)
        q,r = gradient(costfun,X[:,k],U[:,k])
        grad[off .+ ind1.x] = q
        grad[off .+ ind1.u] = r
    end
    grad .*= dt
    off = (N-1)*(n+m)
    qf = gradient(costfun,X[:,N])
    grad[off .+ ind1.x] = qf
    return nothing
end
function gradient(solver::Solver,Z::PrimalVars)
    n,m,N = get_sizes(Z)
    Nx = N*n
    Nu = (N-1)*m
    Nz = Nx + Nu
    grad = spzeros(Nx+Nu)
    gradient!(grad,solver,Z)
    return grad
end

function get_bounds(solver::Solver, equalities::Bool=true)
    Nx,Nu,Nh,Ng,NN = get_batch_sizes(solver)
    n,m,N = get_sizes(solver)
    obj = solver.obj

    x_L = repeat(obj.x_min,1,N)
    x_U = repeat(obj.x_max,1,N)
    u_L = repeat(obj.u_min,1,N-1)
    u_U = repeat(obj.u_max,1,N-1)

    if equalities
        # Initial Condition
        x_L[1:n,1] .= obj.x0
        x_U[1:n,1] .= obj.x0

        # Terminal Constraint
        if obj.use_xf_equality_constraint
            x_L[1:n,N] .= obj.xf
            x_U[1:n,N] .= obj.xf
        end
    end

    # Pack them together
    z_L = PrimalVars(x_L,u_L).Z
    z_U = PrimalVars(x_U,u_U).Z

    return z_L, z_U
end

function get_bound_multipliers(solver::Solver, res::SolverIterResults)
    n,m,N = get_sizes(solver)
    λ = zeros(n+m,N-1)
    x_L = isfinite.(solver.obj.x_min)
    x_U = isfinite.(solver.obj.x_max)
    u_L = isfinite.(solver.obj.u_min)
    u_U = isfinite.(solver.obj.x_min)
    labels = get_constraint_labels(solver)
    inds = (x_U=labels .== "state (upper bound)",
            x_L=labels .== "state (lower bound)",
            u_U=labels .== "control (upper bound)",
            u_L=labels .== "control (lower bound)",
            x=1:n, u=n.+(1:m))
    for k = 1:N-1
        lambda_ = res.λ[k]
        lambda = zeros(n+m,2)
        lambda[inds.x[x_L],1] = lambda_[inds.x_L]
        lambda[inds.x[x_U],2] = lambda_[inds.x_U]
        lambda[inds.u[u_L],1] = lambda_[inds.u_L]
        lambda[inds.u[u_U],2] = lambda_[inds.u_U]
        for i = 1:n+m
            if lambda[i,1] > 0
                λ[i] = -lambda[i,1]
            elseif lambda[i,2] > 0
                λ[i] = lambda[i,2]
            end
        end
    end
end



# Newton Polishing
function newton_snopt(solver::Solver, res::SolverIterResults,tol=1e-10)
    mycost, c, grad_J, hess_J, jacob_c = gen_usrfun_newton(solver)
    Z = PrimalVars(res)

    function usrfun(Z)
        J = mycost(Z)
        gJ = grad_J(Z)

        cineq = Float64[]
        gcineq = Float64[]

        ceq = c.d(Z)
        gceq = jacob_c.d(Z)

        fail = false
        # return J, cineq, ceq, fail
        return J, cineq, ceq, gJ, gcineq, gceq, fail
    end

    lb,ub = get_bounds(solver)
    convertInf!(lb)
    convertInf!(ub)

    options = Dict{String, Any}()
    options["Derivative option"] = 0
    options["Verify level"] = 1
    options["Major feasibility tol"] = tol
    # options["Minor optimality  tol"] = solver.opts.eps_intermediate
    options["Major optimality  tol"] = tol
    sumfile = "logs/snopt-summary.out"
    printfile = "logs/snopt-print.out"

    problem = SnoptProblem(usrfun, Z.Z, lb, ub, options)
    t_eval = @elapsed Zopt,optval,status = Snopt.solve(problem)
    stats = parse_snopt_summary()
    stats["info"] = status
    stats["runtime"] = t_eval

    return Zopt,optval,stats
    Zopt = PrimalVars(Zopt,n,m,N)
    res_snopt = ConstrainedVectorResults(solver,Zopt.X,Zopt.U)
    backwardpass!(res_snopt,solver)
    rollout!(res_snopt,solver,0.0)
    return res_snopt
end

function newton_ipopt(solver::Solver,res::SolverIterResults)
    # Convert to PrimalVar
    Z = PrimalVars(res)

    # Get Constants
    n,m,N = get_sizes(solver)
    Nz = length(Z.Z)   # Number of decision variables
    Nx = N*n           # Number of collocation constraints
    nH = 0             # Number of entries in cost hessian

    # Generate functions
    eval_f, c, grad_J, hess_J, jacob_c = gen_usrfun_newton(solver)

    # Pre-allocate jacobian
    ∇c = jacob_c.d()
    row,col,v = findnz(∇c)
    nG = length(v)  # Number of entries in constraint jacobian

    # Set up functions for Ipopt
    eval_g(Z,g) = c.d(g,Z)
    eval_grad_f(Z, grad_f) = grad_J(grad_f,Z)
    function eval_jac_g(Z, mode, rows, cols, vals)
        if mode == :Structure
            copyto!(rows,row)
            copyto!(cols,col)
        else
            jacob_c.d(∇c,Z)
            copyto!(vals,nonzeros(∇c))
        end
    end

    # Get bounds
    lb,ub = get_bounds(solver)
    convertInf!(lb)
    convertInf!(ub)

    g_L = zeros(Nx)
    g_U = zeros(Nx)
    P = length(g_L)

    # Create Problem
    prob = Ipopt.createProblem(Nz, lb, ub, P,g_L, g_U, nG, nH,
        eval_f, eval_g, eval_grad_f, eval_jac_g)
    prob.x = Z.Z

    # Set options
    dir = root_dir()
    opt_file = joinpath(dir,"ipopt.opt")
    addOption(prob,"option_file_name",opt_file)
    if solver.opts.verbose == false
        addOption(prob,"print_level",0)
    end

    # Solve
    t_eval = @elapsed status = solveProblem(prob)
    stats = parse_ipopt_summary()
    stats["info"] = Ipopt.ApplicationReturnStatus[status]
    stats["runtime"] = t_eval
    return prob.x, stats

    # Convert back to feasible trajectory
    vars = PrimalVars(prob.x,n,m,N)
    res_ipopt = ConstrainedVectorResults(solver,Zopt.X,Zopt.U)
    backwardpass!(res_snopt,solver)
    rollout!(res_snopt,solver,0.0)
    return res_ipopt
end


function gen_newton_functions(solver::Solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
    n,m,N = get_sizes(solver)

    obj = solver.obj
    costfun = obj.cost


    Nx,Nu,Nh,Ng,NN = get_batch_sizes(solver)
    Nz = Nx+Nu

    ind_x, ind_u = get_primal_inds(n,m,N)

    names = (:z,:ν,:λ,:μ)
    ind1 = TrajectoryOptimization.create_partition([Nz,Nx,Nh,Ng],names)
    ind2 = TrajectoryOptimization.create_partition2([Nz,Nx,Nh,Ng],names)
    ind1 = merge(ind1,create_partition([Nx,Nu],(:x,:u)))
    ind1 = merge(ind1,create_partition([Nz,Nx+Nh+Ng],(:z,:y)))
    ind2 = merge(ind2,create_partition2([Nz,Nx+Nh+Ng],(:z,:y)))

    mycost, c, grad_J, hess_J, jacob_c = gen_usrfun_newton(solver)

    Z0 = PrimalVars(n,m,N)
    ∇²J = hess_J(Z0.Z)

    function active_set(Z,eps=1e-3)
        a = ones(Bool,NN)
        ci = c.I(Z)
        c_inds = ones(Bool,length(ci))
        c_inds = ci .>= -eps
        a[ind_v.μ] = c_inds
        return a
    end

    function active_set(V::NewtonVars,eps=1e-3)
        ci = c.I(primals(V))
        c_inds = ones(Bool,length(ci))
        c_inds = ci .>= -eps
        V.a[ind1.μ] = c_inds
    end

    function buildKKT(V::NewtonVars,ρ,type=:kkt)
        Z = primals(V)
        A = spzeros(NN,NN)
        A[ind2.zz...] = ∇²J
        D = view(A,ind2.νz...)
        H = view(A,ind2.λz...)
        G = view(A,ind2.μz...)
        jacob_c.d(D,Z)
        jacob_c.E(H,Z)
        jacob_c.I(G,Z)
        A = Symmetric(A')

        b = zeros(NN)
        d = view(b,ind1.ν)
        h = view(b,ind1.λ)
        g = view(b,ind1.μ)
        c.d(d,Z)
        c.E(h,Z)
        c.I(g,Z)
        return A,b
    end

    #
    # function constraint_projection!(A,b,V::NewtonVars)
    #     Z = primals(V)
    #     D = view(A,ind2.νz)
    #     H = view(A,ind2.λz)
    #     G = view(A,ind2.μz)
    #     jacob_c.d(D,Z)
    #     jacob_c.E(H,Z)
    #     jacob_c.I(G,Z)
    #
    #     d = view(b,ind1.ν)
    #     h = view(b,ind1.λ)
    #     g = view(b,ind1.μ)
    #     c.d(d,Z)
    #     c.E(h,Z)
    #     c.I(g,Z)
    #
    #     y = view(b,ind1.y)
    #     Y = view(A,ind2.yz)
    #     δV̂ = -Y'*((Y*Y')\y)
    #     return δV̂
    # end
    #
    function newton_step(V::NewtonVars,ρ; type=:kkt, eps=1e-2, reg=Diagonal(zeros(length(V))),verbose=solver.opts.verbose)
        max_c2(V) = max(Ng > 0 ? maximum(c.I(V.V)) : 0, norm(c.E(V.V),Inf))
        max_c(V) = max(norm(c.d(V.V),Inf),max_c2(V))

        meritfun = mycost

        V_ = copy(V)
        active_set(V,eps)
        amu = V.a[ind1.μ]
        Z_ = primals(V_)

        d1 = c.d(Z_)
        h1 = c.E(Z_)
        g1 = c.I(Z_)
        y = [d1;h1;g1[amu]]
        δz = zero(ind1.z)
        if verbose; println("max y: $(maximum(abs.(y)))") end
        while norm(y,Inf) > 1e-6
            D = jacob_c.d(Z_)
            H = jacob_c.E(Z_)
            G = jacob_c.I(Z_)
            Y = [D;H;G[amu,:]]

            #δV̂ = -(∇²J\(Y'))*(Y*(∇²J\(Y'))\y)
            δZ = -Y'*((Y*Y')\y)
            Z_ .+= δZ

            d1 = c.d(Z_)
            h1 = c.E(Z_)
            g1 = c.I(Z_)
            y = [d1;h1;g1[amu]]
            if verbose; println("max y: $(maximum(abs.(y)))") end
        end

        J0 = mycost(V_)
        if verbose; println("Initial Cost: $J0") end

        # Build and solve KKT
        A,b = buildKKT(V_,ρ,type)
        active_set(V_,eps)
        a = V_.a
        amu = a[ind1.μ]
        δV = zero(V.V)
        Ā = A[a,a] + reg[a,a]
        δV[a] = -Ā\b[a]

        # Line Search
        ϕ=0.01
        α = 2
        V1 = copy(V_)
        Z1 = primals(V1)
        δV1 = α.*δV
        J = J0+1e8
        while J > J0 #+ α*ϕ*b'δV1
            α *= 0.5
            δV1 = α.*δV
            V1.V  .= V_.V + δV1

            d1 = c.d(Z1)
            h1 = c.E(Z1)
            g1 = c.I(Z1)
            y = [d1;h1;g1[amu]]
            if verbose; println("max y: $(maximum(abs.(y)))") end
            while norm(y,Inf) > 1e-6
                D = jacob_c.d(Z1)
                H = jacob_c.E(Z1)
                G = jacob_c.I(Z1)
                Y = [D;H;G[amu,:]]

                #δV̂ = -(∇²J\(Y'))*(Y*(∇²J\(Y'))\y)
                δZ = -Y'*((Y*Y')\y)
                Z1 .+= δZ

                d1 = c.d(Z1)
                h1 = c.E(Z1)
                g1 = c.I(Z1)
                y = [d1;h1;g1[amu]]
                if verbose; println("max y: $(maximum(abs.(y)))") end
            end

            J = meritfun(V1)
            if verbose; println("New Cost: $J") end
        end

        # Multiplier projection
        ∇J = grad_J(Z1)
        d1 = c.d(Z1)
        h1 = c.E(Z1)
        g1 = c.I(Z1)
        y = [d1;h1;g1[amu]]
        D = jacob_c.d(Z1)
        H = jacob_c.E(Z1)
        G = jacob_c.I(Z1)
        Y = Array([D;H;G[amu,:]])

        ν = V1.ν
        μ = V1.μ
        λ = V1.λ
        lambda = [ν;λ;μ[amu]]
        r = ∇J + Y'lambda
        δlambda = zero(lambda)
        if verbose; println("max residual before: $(norm(r,Inf))") end
        δlambda -= (Y*Y')\(Y*r)
        lambda1 = lambda + δlambda
        r = ∇J + Y'lambda1
        if verbose; println("max residual after: $(norm(r,Inf))") end
        ν1 = lambda1[1:Nx]
        λ1 = lambda1[Nx .+ (1:Nh)]
        μ1 = lambda1[(Nx+Nh) .+ (1:count(amu))]
        V1.ν .= ν1
        V1.λ .= λ1
        V1.μ[amu] = μ1
        J = meritfun(V1)
        if verbose; println("New Cost: $J") end

        # Take KKT Step
        cost = J
        A,b = buildKKT(V1,ρ,type)
        active_set(V1,eps)
        a = V1.a
        grad = norm(b[a])
        c_max = max_c(V1)

        change(x,x0) = format((x0-x)/x0*100,precision=4) * "%"
        if verbose
            println()
            println("  cost: $J $(change(J,J0))")
            println("  step: $(norm(δV1))")
            println("  grad: $(grad)")
            println("  c_max: $(c_max)")
            println("  c_max2: $(max_c2(V1))")
            println("  α: $α")
            # println("  rank: $(rank(Ā))")
            # println("  cond: $(cond(Ā))")
        end
        stats = Dict("cost"=>cost,"grad"=>grad,"c_max"=>c_max)
        return V1
    end

    return newton_step, buildKKT, active_set
end

function newton_projection(solver::Solver,res::SolverIterResults; kwargs...)
    V = NewtonVars(solver,res)
    newton_step, = gen_newton_functions(solver)
    V_ = newton_step(V,0; kwargs...)
end
