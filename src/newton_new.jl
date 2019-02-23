
struct NewtonSolver
    cost::Function
    dynamics::Function
    cE::Function
    cI::Function

    ind1::NamedTuple
    ind2::NamedTuple
    sizes::NamedTuple
    pack::Function
    unpack::Function
    opts::Dict{Symbol,Any}
end


function NewtonSolver(solver::Solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
    n,m,N = get_sizes(solver)

    c_function!, c_jacobian!, c_labels, cI!, cE! = generate_constraint_functions(solver.obj)

    Nx = N*n
    Nu = (N-1)*m
    Nz = Nx + Nu
    Nh = (N-1)*pE + pE_N
    Ng = (N-1)*pI + pI_N
    NN = 2Nx + Nu + Nh + Ng

    names = (:z,:ν,:λ,:μ)
    ind1 = TrajectoryOptimization.create_partition([Nz,Nx,Nh,Ng],names)
    ind2 = TrajectoryOptimization.create_partition2([Nz,Nx,Nh,Ng],names)
    indx = reshape(1:Nx,n,N)
    indu = reshape(1:Nu,m,N-1) .+ Nx

    sizes = (Nx=Nx,Nu=Nu,Nz=Nz,Nh=Nh,Ng=Ng,NN=NN,pI=pI,pE=pE,pI_N=pI_N,pE_N=pE_N,n=n,m=m,N=N,
        x=indx, u=indu)


    function mycost(Z)
        X,U = unpackZ(Z,sizes)
        cost(solver,X,U)
    end

    function cE(Z)
        X,U = unpackZ(Z,sizes)

        C = zeros(eltype(Z),pE,N-1)
        for k = 1:N-1
            cE!(view(C,1:pE,k),X[:,k],U[:,k])
        end
        CN = zeros(eltype(Z),pE_N)
        cE!(CN,X[:,N])
        CE = [vec(C); CN]
        return CE
    end

    function cI(Z)
        X,U = unpackZ(Z,sizes)

        C = zeros(eltype(Z),pI,N-1)
        for k = 1:N-1
            cI!(view(C,1:pI,k),X[:,k],U[:,k])
        end
        CN = zeros(eltype(Z),pI_N)
        cI!(CN,X[:,N])
        CI = [vec(C); CN]
        return CI
    end

    function dynamics(Z)
        X,U = unpackZ(Z,sizes)

        D = zeros(eltype(Z),n,N)
        D[:,1] = X[:,1] - solver.obj.x0
        for k = 2:N
            solver.fd(view(D,1:n,k),X[:,k-1],U[:,k-1])
            D[:,k] -= X[:,k]
        end
        return vec(D)
    end

    function unpack(V::AbstractVector)
        if length(V) == NN
            Z = view(V,ind1.z)
            ν = view(V,ind1.ν)
            λ = view(V,ind1.λ)
            μ = view(V,ind1.μ)
            return Z,ν,λ,μ
        elseif length(V) == Nz
            X = V[indx]
            U = V[indu]
        end
    end

    function pack(Z::Vector,ν::Vector,λ::Vector,μ::Vector)
        V = zeros(NN)
        V[ind1.z] = Z
        V[ind1.ν] = ν
        V[ind1.λ] = λ
        V[ind1.μ] = μ
        return V
    end

    ix = 1:Nx
    iu = Nx .+ (1:Nu)
    function pack(X::Matrix,U::Matrix)
        Z = zeros(Nz)
        Z[ix] = vec(X)
        Z[iu] = vec(U)
        return Z
    end
    pack(X::Matrix, U::Matrix, ν::T, λ::T, μ::T) where T <: Vector = pack(pack(X,U),ν,λ,μ)

    opts = Dict(:type=>:kkt,                # Type of hessian to assemble
                :ϵ_as=>1e-3,                # Active set tolerance
                :iters_linesearch=>10,      # Number of iterations to use in the linesearch
                :refinement=>:jacobian,     # Which refinement method to use
                :η=>0.1                     # Gradient descent criteria
                )

    NewtonSolver(mycost,dynamics,cE,cI,
        ind1,ind2,sizes,pack,unpack,opts)
end

function get_sizes(solver::NewtonSolver)
    solver.sizes.n, solver.sizes.m, solver.sizes.N
end

function get_num_constraints(solver::NewtonSolver)
    pI = solver.sizes.pI
    pE = solver.sizes.pE
    return pI+pE,pI,pE
end

function get_num_terminal_constraints(solver::NewtonSolver)
    pI = solver.sizes.pI_N
    pE = solver.sizes.pE_N
    return pI+pE,pI,pE
end

function max_violation(solver::NewtonSolver,V::AbstractVector,use_dynamics::Bool=true)
    c_max = max(solver.sizes.Ng > 0 ? maximum(solver.cI(V)) : 0, norm(solver.cE(V),Inf))
    if use_dynamics
        c_max = max(norm(solver.dynamics(V),Inf),c_max)
    end
    return c_max
end

function unpackZ(Z::AbstractVector, inds::NamedTuple)
    X = view(Z,inds.x)
    U = view(Z,inds.u)
    return X,U
end
unpackZ(Z::AbstractVector, solver::NewtonSolver) = unpackZ(Z,solver.sizes)

function create_V(solver::NewtonSolver,res::SolverIterResults)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
    n,m,N = get_sizes(solver)

    x = to_array(res.X)
    u = to_array(res.U)
    ν = zeros(solver.sizes.Nx)

    if res isa ConstrainedVectorResults
        μ = zeros(pI,N-1)
        λ = zeros(pE,N-1)
        ineq = 1:pI
        eq = pI .+ (1:pE)
        for k = 1:N-1
            μ[:,k] = res.λ[k][ineq]
            λ[:,k] = res.λ[k][eq]
        end
        μ = vec(μ)
        λ = vec(λ)
        append!(μ, res.λ[N][1:pI_N])
        append!(λ, res.λ[N][pI_N .+ (1:pE_N)])
    else
        λ = Float64[]
        μ = Float64[]
    end
    return solver.pack(x,u,ν,λ,μ)
end


function gen_newton_functions(solver::NewtonSolver)
    ind1 = solver.ind1
    ind2 = solver.ind2

    cE,cI,dynamics = solver.cE, solver.cI, solver.dynamics
    mycost = solver.cost

    sizes = solver.sizes
    Nx = sizes.Nx
    Nz = sizes.Nz
    Nh = sizes.Nh
    Ng = sizes.Ng

    function buildKKT(V, ρ, type)
        Z, ν, λ, μ = solver.unpack(V)
        X,U = unpackZ(Z, solver)

        ∇²J = ForwardDiff.hessian(mycost,Z)
        ∇J = ForwardDiff.gradient(mycost,Z)
        D = ForwardDiff.jacobian(dynamics,Z)
        H = ForwardDiff.jacobian(cE,Z)
        G = ForwardDiff.jacobian(cI,Z)
        d = dynamics(Z)
        h = cE(Z)
        g = cI(Z)

        if type == :penalty
            A = [∇²J   D'         H'      G';
                 D   -1/ρ*I       zeros(Nx,Nh+Ng) ;
                 H   zeros(Nh,Nx) -1/ρ*I zeros(Nh,Ng);
                 G   zeros(Ng,Nx) zeros(Ng,Nh) -1/ρ*I]
            b = [∇J + D'nu + H'λ + G'μ; d; h; g]
        elseif type == :kkt
            A = [∇²J   D'    H'                G';
                 D   zeros(Nx,Nx) zeros(Nx,Nh) zeros(Nx,Ng);
                 H   zeros(Nh,Nx) zeros(Nh,Nh) zeros(Nh,Ng);
                 G   zeros(Ng,Nx) zeros(Ng,Nh) zeros(Ng,Ng)]
            b = [∇J + D'ν + H'λ + G'μ; d; h; g]
        end
        return A,b
    end

    a = ones(Bool,solver.sizes.NN)
    function active_set(Z,eps=0)
        X,U = unpackZ(Z, solver)
        ci = cI(Z)
        c_inds = ones(Bool,solver.sizes.Ng)
        c_inds = ci .>= -eps

        a[ind1.μ] = c_inds
        return a
    end

    function newton_step(V,ρ,meritfun::Function,type=:kkt; projection=:none,
        reg=Diagonal(zero(V)), η=solver.opts[:η])

        # Initial cost
        J0 = meritfun(V)
        V_ = copy(V)

        # Build and solve KKT
        A,b = buildKKT(V_,ρ,type)
        a = active_set(V_,1e-3)
        amu = a[ind1.μ]
        δV = zero(V)
        Ā = A[a,a] + reg[a,a]
        δV[a] = -Ā\b[a]

        if projection == :jacobian
            V1 = V + δV
            Z1 = V1[ind1.z]
            Z = V[ind1.z]
            D = ForwardDiff.jacobian(dynamics,Z)
            H = ForwardDiff.jacobian(cE,Z)
            G = ForwardDiff.jacobian(cI,Z)
            Y = [D;H;G[amu,:]]
            d1 = dynamics(Z1)
            h1 = cE(Z1)
            g1 = cI(Z1)
            y = [d1;h1;g1[amu]]
            # dv̂ = -D'*((D*D')\d1)
            δV̂ = -Y'*((Y*Y')\y)
            δV[ind1.z] += δV̂
        elseif projection == :constraints
            b̂ = copy(b)
            for j = 1:5
                V1 = V_ + δV
                d1 = dynamics(V1)
                h1 = cE(V1)
                g1 = cI(V1)
                b̂[ind1.ν] = d1
                b̂[ind1.λ] = h1
                b̂[ind1.μ] = g1
                δV[a] = -Ā\b̂[a]
                V_ = V_ + δV
            end
        end

        # Line Search
        α = armijo_line_search(meritfun,V,δV,b, max_iter=solver.opts[:iters_linesearch])
        V_ = V_ + α*δV

        stats = (α=α,condnum=cond(Ā),grad=norm(b))
        return V_, stats
    end

    return newton_step, buildKKT, active_set
end

function armijo_line_search(merit::Function,V,d,grad; max_iter=10, ϕ=0.01)
    α = 1
    J_prev = merit(V)
    J = merit(V+α*d)
    iter = 1
    while J > J_prev + α*ϕ*grad'd
        α *= 0.75
        J = merit(V + α*d)
        if iter > max_iter
            println("Max iterations")
            α = 0
            break
        end
        iter += 1
    end
    return α
end

function meritfunction(solver::NewtonSolver,V,ρ)
    eps = solver.opts[:ϵ_as]
    Z, ν, λ, μ = solver.unpack(V)
    d = solver.dynamics(Z)
    h = solver.cE(Z)
    g = solver.cI(Z)
    a = g .> eps
    solver.cost(Z) + ν'd + λ'h + μ'g + ρ/2*sqrt(d'd + h'h + g[a]'g[a])
end

function solve(solver::NewtonSolver,V::Vector, ρ=0; iters=1, verbose=true, meritfun=V->meritfunction(solver,V,ρ), projection=:none)

    newton_step, buildKKT, active_set = gen_newton_functions(solver)
    change(x,x0) = format((x0-x)/x0*100,precision=2) * "%"

    # Stats
    cost = zeros(iters)
    grad = zeros(iters)
    c_max = zeros(iters)
    c_max0 = zeros(iters)  # max violation without dynamics

    J_prev = meritfun(V)
    for i = 1:iters
        V, step_stats = newton_step(V,ρ,meritfun,projection=projection)
        J = meritfun(V)
        grad[i] = step_stats.grad
        c_max[i] = max_violation(solver,V)
        c_max0[i] = max_violation(solver,V,false)

        if verbose
            println("iter $i")
            println("  cost: $J $(change(J,J_prev))")
            println("  grad: $(grad[i])")
            println("  c_max: $(c_max[i])")
            println("  c_max0: $(c_max0[i]))")
            println("  α: $(step_stats.α)")
            println("  cond: $(step_stats.condnum)")
            println()
        end
        J_prev = J
    end
    return V
end
function solve(solver::NewtonSolver,res::ConstrainedIterResults; kwargs...)
    ρ = max_penalty(res)
    meritfun(V) = meritfunction(solver,V,ρ)
    solve(solver,create_V(solver,res),ρ)
end



# model, obj = Dynamics.pendulum
# solver = Solver(model, obj, N=21)
# res, stats = solve(solver)
#
# nsolver = NewtonSolver(solver)
# meritfun(V) = meritfunction(nsolver,V,0)
# solve(nsolver,res,iters=2,meritfun=meritfun)
