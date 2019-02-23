import Base.getindex

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


function gen_usrfun_newton(solver::Solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
    n,m,N = get_sizes(solver)

    c_function!, c_jacobian!, c_labels, cI!, cE! = generate_constraint_functions(solver.obj)
    costfun = solver.obj.cost

    Nx = N*n
    Nu = (N-1)*m
    Nz = Nx + Nu
    Nh = (N-1)*pE + pE_N
    Ng = (N-1)*pI + pI_N
    NN = 2Nx + Nu + Nh + Ng

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


    # Get integration scheme
    if isdefined(TrajectoryOptimization,solver.integration)
        discretizer = eval(solver.integration)
    else
        throw(ArgumentError("$integration is not a defined integration scheme"))
    end
    f!(xdot,x,u) = TrajectoryOptimization.dynamics(solver.model,xdot,x,u)
    fd! = rk4(f!,solver.dt)
    fd_aug!(xdot,z) = fd!(xdot,view(z,ind_z.x),view(z,ind_z.u))
    Fd!(dz,xdot,z) = ForwardDiff.jacobian!(dz,fd_aug!,xdot,z)

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

    grad_J(grad,Z) = gradient!(grad,solver,PrimalVars(Z))
    grad_J(Z) = gradient(solver,PrimalVars(Z,ind_x,ind_u))

    hess_J(hess,Z) = copyto!(hess,∇²J)
    hess_J(Z) = ∇²J

    function jacob_c(jacob,Z)
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
    function jacob_c(Z)
        jacob = spzeros(Nx,Nz)
        jacob_c(jacob,Z)
        return jacob
    end

    return mycost, dynamics, cI, cE, grad_J, hess_J, jacob_c
end

"""SIGNATURES)
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
    @show size(grad)
    gradient!(grad,solver,Z)
    return grad
end


model,obj = Dynamics.dubinscar_parallelpark
solver = Solver(model,obj,N=21)
n,m,N = get_sizes(solver)
p,pI,pE = get_num_constraints(solver)
p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
obj.cost.Q .= Diagonal(I,n)
obj.cost.R .= Diagonal(I,m)*2
obj.cost.H .= ones(m,n)

Nx = N*n
Nu = (N-1)*m
Nz = Nx+Nu
X = rand(n,N)
U = rand(m,N-1)
Z = packZ(X,U)

V = [Z; rand(21)]
P = PrimalVars(V,n,m,N)
P.Z == Z
P.X == X
P.U == U
get_sizes(P) == (n,m,N)
P[3] == [X[:,3]; U[:,3]]

Nx = N*n
Nu = (N-1)*m
Nz = Nx + Nu
Nh = (N-1)*pE + pE_N
Ng = (N-1)*pI + pI_N
NN = 2Nx + Nu + Nh + Ng

names = (:z,:ν,:λ,:μ)
ind1 = TrajectoryOptimization.create_partition([Nz,Nx,Nh,Ng],names)
ind2 = TrajectoryOptimization.create_partition2([Nz,Nx,Nh,Ng],names)

ind_x, ind_u = get_primal_inds(n,m,N)
ind_z = TrajectoryOptimization.create_partition([Nx,Nu],(:x,:u))
ind_h = create_partition([(N-1)*pE,pE_N],(:k,:N))
ind_g = create_partition([(N-1)*pI,pI_N],(:k,:N))
ind_h = (k=ind_h.k, N=ind_h.N, k_mat=reshape(ind_h.k,pE,N-1))
ind_g = (k=ind_g.k, N=ind_g.N, k_mat=reshape(ind_g.k,pI,N-1))

CE = zeros(Nh)
CE[ind_h.k_mat]
CI = zeros(Ng)
CI[ind_g.k_mat]

Z = packZ(X,U)
mycost, dyn, cI, cE, grad_J, hess_J, jacob_c = gen_usrfun_newton(solver)
mycost(Z)
dyn(Z)
cI(Z)
cE(Z)
grad_J(Z) == ForwardDiff.gradient(mycost,Z)
Array(jacob_c(Z)) == ForwardDiff.jacobian(dyn,Z)
