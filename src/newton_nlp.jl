import Base.getindex
using Snopt

struct PrimalVars{T}
    Z::Vector{T}
    X::SubArray{T}
    U::SubArray{T}
end

function PrimalVars(Z::Vector,ind_x::Matrix{Int},ind_u::Matrix{Int})
    n,N = size(ind_x)
    m = size(ind_u,1)
    Nz = N*n + (N-1)*m
    Z = Z[1:Nz]
    X = view(Z,ind_x)
    U = view(Z,ind_u)
    PrimalVars(Z,X,U)
end

function PrimalVars(Z::Vector,n::Int,m::Int,N::Int)
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

function ConstrainedVectorResults(Z::PrimalVars)


get_sizes(Z::PrimalVars) = size(Z.X,1), size(Z.U,1), size(Z.X,2)

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
    Z::PrimalVars{T}
    ν::Vector{T}
    λ::Vector{T}
    μ::Vector{T}
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
    z_bnd = [x_bnd; u_bnd]
    active_bnd = z_bnd .!= 0
    jac_x_bnd= [Diagonal(isfinite.(obj.x_max));
               -Diagonal(isfinite.(obj.x_min))]
    jac_u_bnd = [Diagonal(isfinite.(obj.u_max));
                -Diagonal(isfinite.(obj.u_min))]
    jac_bnd_k = blockdiag(jac_x_bnd,jac_u_bnd)[active_bnd,:]
    jac_bnd = blockdiag([jac_bnd_k for k = 1:N-1]...)
    jac_bnd = blockdiag(jac_bnd,jac_x_bnd[active_bnd[1:2n],:])

    function mycost(Z)
        Z̄ = PrimalVars(Z,ind_x,ind_u)
        X,U = Z̄.X, Z̄.U
        cost(solver,X,U)
    end

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
    jacob_cI(jacob,Z) = copyto(jacob,jac_bnd)
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

# Set up Problem
model,obj = Dynamics.dubinscar
obj.cost.Q .= Diagonal(I,3)*1e-1
obj.cost.R .= Diagonal(I,2)*1e-1
obj.cost.Qf .= Diagonal(I,3)*100
obj_c = ConstrainedObjective(obj,u_max=0.75,u_min=-0.75,x_min=[-0.5;-0.1;-Inf],x_max=[0.5,1.1,Inf])

# iLQR Solution
solver = Solver(model,obj_c,N=21)
n,m,N = get_sizes(solver)
U0 = ones(m,N-1)
solver.opts.verbose = true
res,stats = solve(solver,U0)
plot()
plot_trajectory!(res)
plot(res.X)


# Newton Polishing
function newton_snopt(solver::Solver, res::SolverIterResults)
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
    options["Major feasibility tol"] = 1e-10
    # options["Minor optimality  tol"] = solver.opts.eps_intermediate
    options["Major optimality  tol"] = 1e-10
    sumfile = "logs/snopt-summary.out"
    printfile = "logs/snopt-print.out"

    Zopt,optval,status = snopt(usrfun, Z.Z, lb, ub, options)
    Zopt = PrimalVars(Zopt,n,m,N)
    res_snopt = ConstrainedVectorResults(solver,Zopt.X,Zopt.U)
    backwardpass!(res_snopt,solver)
    rollout!(res_snopt,solver,0.0)
    return res_snopt
end
res_snopt = newton_snopt(solver,res)
max_violation(res_snopt)

mycost, c, grad_J, hess_J, jacob_c = gen_usrfun_newton(solver)
∇c = jacob_c.d()
jacob_c.d(∇c,Z.Z)
∇c
nonzeros(∇c)

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

    # Convert back to feasible trajectory
    vars = PrimalVars(prob.x,n,m,N)
    res_ipopt = ConstrainedVectorResults(solver,Zopt.X,Zopt.U)
    backwardpass!(res_snopt,solver)
    rollout!(res_snopt,solver,0.0)
    return res_ipopt
end

solver.opts.verbose = false
res_ipopt = newton_ipopt(solver,res)
max_violation(res_ipopt)
