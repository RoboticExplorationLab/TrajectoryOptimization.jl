include("partitioning.jl")
function gen_newton_functions(solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    c_function!, c_jacobian!, c_labels, cI!, cE! = generate_constraint_functions(solver.obj)
    f = solver.fd

    Nz,Ng,Nh,Nx = get_batch_sizes2(solver)
    NN = Nz+Ng+Ng+Nh+Nx

    names = (:z,:s,:μ,:λ,:ν)
    ind1 = create_partition([Nz,Ng,Ng,Nh,Nx],names)
    ind2 = create_partition2([Nz,Ng,Ng,Nh,Nx],names)
    indx = 1:N*n
    indu = N*n .+ (1:(N-1)*m)
    s_x = reshape(1:(N*n),n,N)
    s_u = reshape(N*n .+ (1:(m*(N-1))),m,N-1)

    function objective_cost(Z::Vector)
        X,U = unpackZZ(Z)
        J = cost(solver,X,U)
    end

    function al_cost(V::Vector,ρ=1000)
        X,U,s,μ,λ,ν = unpackV(V)
        e = cE(V)
        i = cI(V) + 0.5s.^2
        dd = d(V)
        J = cost(solver,X,U) + λ'e + μ'i + ν'dd + 0.5*ρ*(e'e + i'i + dd'dd)
        return J
    end

    function cI(Z)
        X,U = unpackZZ(Z)
        C = zeros(eltype(Z),pI,N-1)
        for k = 1:N-1
            cI!(view(C,1:pI,k),X[:,k],U[:,k])
        end
        CN = zeros(eltype(Z),pI_N)
        cI!(CN,X[:,N])
        CI = [vec(C); CN]
        return CI
    end

    function cE(Z)
        X,U = unpackZZ(Z)
        C = zeros(eltype(Z),pE,N-1)
        for k = 1:N-1
            cE!(view(C,1:pE,k),X[:,k],U[:,k])
        end
        CN = zeros(eltype(Z),pE_N)
        cE!(CN,X[:,N])
        CE = [vec(C); CN]
        return CE
    end

    function d(Z)
        X,U = unpackZZ(Z)
        D = zeros(eltype(Z),n,N)
        D[:,1] = X[:,1] - solver.obj.x0
        for k = 2:N
            f(view(D,1:n,k),X[:,k-1],U[:,k-1])
            D[:,k] -= X[:,k]
        end
        return vec(D)
    end

    function calculate_jacobians(Z::Vector)
        ∇²J = ForwardDiff.hessian(objective_cost,Z)
        ∇J = ForwardDiff.gradient(objective_cost,Z)
        G = ForwardDiff.jacobian(cI,Z)
        H = ForwardDiff.jacobian(cE,Z)
        D = ForwardDiff.jacobian(d,Z)
        return ∇²J,∇J, G,H,D
    end

    function active_set(Z,eps=1e-2)
        X,U = unpackZZ(Z)
        a = ones(Bool,NN)
        ci = cI(Z)
        c_inds = ones(Bool,length(ci))
        c_inds = ci .>= -eps

        a[ind1.μ] = c_inds
        a[ind1.s] = c_inds
        return a
    end


    function packV(X,U,s,μ,λ,ν)
        V = [vec(X);vec(U);s;μ;λ;ν]
    end

    function unpackV(V)
        X = V[s_x]
        U = V[s_u]
        s = V[ind1.s]
        μ = V[ind1.μ]
        λ = V[ind1.λ]
        ν = V[ind1.ν]
        return X,U,s,μ,λ,ν
    end

    function packZZ(X,U)
        Z = [vec(X);vec(U)]
    end

    function unpackZZ(Z)
        X = Z[s_x]
        U = Z[s_u]
        return X,U
    end

    function buildKKT(V,ρ=1000,type=:regularized,reg=0)
        X,U,s,μ,λ,ν = unpackV(V)
        Z = packZZ(X,U)
        ∇²J,∇J,G,H,D = calculate_jacobians(Z)
        g = cI(Z)
        h = cE(Z)
        dd = d(Z)

        A = spzeros(NN,NN)
        b = zeros(NN)

        if type == :regularized
            A[ind2.zz] = ∇²J
            A[ind2.zμ] = G'
            A[ind2.zλ] = H'
            A[ind2.zν] = D'

            A[ind2.μz] = G
            A[ind2.λz] = H
            A[ind2.νz] = D

            A[ind2.ss] = Diagonal(μ)
            A[ind2.sμ] = Diagonal(s)
            A[ind2.μs] = Diagonal(s)

            A[ind2.μμ] = -1/ρ*Diagonal(I,Ng)
            A[ind2.λλ] = -1/ρ*Diagonal(I,Nh)
            A[ind2.νν] = -1/ρ*Diagonal(I,Nx)

            b[ind1.z] = ∇J + G'μ + H'λ + D'ν
            b[ind1.s] = Diagonal(s)*μ
            b[ind1.μ] = g+s.^2/2
            b[ind1.λ] = h
            b[ind1.ν] = dd
        else
            A[ind2.zz] = ∇²J + ρ*(G'G + H'H + D'D)
            A[ind2.zs] = ρ*G'Diagonal(s)
            A[ind2.zμ] = G'
            A[ind2.zλ] = H'
            A[ind2.zν] = D'

            A[ind2.sz] = ρ*Diagonal(s)*G
            A[ind2.μz] = G
            A[ind2.λz] = H
            A[ind2.νz] = D

            A[ind2.ss] = Diagonal(μ) + ρ*Diagonal(g) + 3/2*Diagonal(s.^2)
            A[ind2.sμ] = Diagonal(s)
            A[ind2.μs] = Diagonal(s)

            A[ind2.μμ] = -reg*Diagonal(I,Ng)
            A[ind2.λλ] = -reg*Diagonal(I,Nh)
            A[ind2.νν] = -reg*Diagonal(I,Nx)

            b[ind1.z] = ∇J + G'μ + H'λ + D'ν + ρ*(H'h + G'*(g+0.5s.^2) + D'dd)
            b[ind1.s] = Diagonal(s)*μ + μ.*s + ρ*g.*s + ρ/2*s.^3
            b[ind1.μ] = g+s.^2/2
            b[ind1.λ] = h
            b[ind1.ν] = dd

        end

        return A,b
    end

    function getV(results::SolverIterResults)
        X = to_array(results.X)
        U = to_array(results.U)

        s = zeros(pI,N-1)
        μ = zeros(pI,N-1)
        λ = zeros(pE,N-1)
        ν = zeros(n,N-1)

        ineq = 1:pI
        eq = pI .+ (1:pE)

        for k = 1:N-1
            s[:,k] = sqrt.(2.0*max.(0,-results.C[k][ineq]))
            μ[:,k] = results.λ[k][ineq]
            λ[:,k] = results.λ[k][eq]
            ν[:,k] = results.s[k]
        end
        s = vec(s)
        μ = vec(μ)
        λ = vec(λ)
        ν = vec(ν)
        append!(s, sqrt.(2.0*max.(0,-results.C[N][1:pI_N])))
        append!(μ, results.λ[N][1:pI_N])
        append!(λ, results.λ[N][pI_N .+ (1:pE_N)])
        append!(ν, results.s[N])
        @show length(s), length(μ), length(λ), length(ν)

        V = packV(X,U,s,μ,λ,ν)
    end

    return buildKKT, getV, packV, unpackV, packZZ, unpackZZ, al_cost, cI, cE, d, active_set, calculate_jacobians, ind1, ind2
end

function get_batch_sizes2(solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    Nz = n*N + m*(N-1)
    Ng = pI*(N-1) + pI_N
    Nh = pE*(N-1) + pE_N
    Nx = N*n
    return Nz,Ng,Nh,Nx
end

function armijo_line_search(merit::Function,V,d,grad; max_iter=10, ϕ=0.01)
    α = 1
    J_prev = merit(V)
    J = Inf
    iter = 1
    while J > J_prev + α*ϕ*grad'd
        α *= 0.75
        J = merit(V - α*d)
        if iter > max_iter
            println("Max iterations")
            break
        end
        iter += 1
    end
    return α
end

model,obj = Dynamics.dubinscar_parallelpark
# obj_c = ConstrainedObjective(obj,u_min=-0.3,u_max=0.4)
solver = Solver(model,obj,N=21)
n,m,N = get_sizes(solver)
U0 = ones(m,N-1)
solver.opts.verbose = true
res,stats = solve(solver,U0)
cost(solver,res)
max_violation(res)
λ_update_default!(res,solver)

buildKKT, getV, packV, unpackV, packZZ, unpackZZ, al_cost, cI, cE, d, active_set, calculate_jacobians, ind1, ind2 = gen_newton_functions(solver::Solver)
max_c(V) = max(maximum(cI(V)),norm(cE(V),Inf),norm(d(V),Inf))

ρ = 1e6
merit(V) = al_cost(V,ρ)
V = getV(res)
J0 = merit(V)
c_max0 = max_c(V)

J_prev = merit(V)
A,b = buildKKT(V,ρ,:new)
a = active_set(V,1)
rank(Array(A))
cond(Array(A))

reg = 1
δV = zero(V)
δV[a] = A[a,a]\b[a]
δV = (A+reg*I)\b
α = armijo_line_search(merit,V,δV,b)
V1 = V - δV*α
@show J_prev - merit(V1)
@show max_c(V) / max_c(V1)
max_c(V1)
V = copy(V1)



# Dubinscar
