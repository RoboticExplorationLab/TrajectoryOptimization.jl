
function gen_newton_functions(solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    c_function!, c_jacobian!, c_labels, cI!, cE! = generate_constraint_functions(solver.obj)
    f = solver.fd
    s_x = reshape(1:(N*n),n,N)
    s_u = reshape(N*n .+ (1:(m*(N-1))),m,N-1)
    s_λ = (N*n + (N-1)*m) .+ (1:((N-1)*pE + pE_N))
    s_μ = (N*n + (N-1)*m + length(s_λ)) .+ (1:((N-1)*pI + pI_N))
    s_ν = (N*n + (N-1)*m + length(s_λ) + length(s_μ)) .+ (1:N*n)
    s_s = (N*n + (N-1)*m + length(s_λ) + length(s_μ) + N*n) .+ (1:length(s_μ))
    sze = (X=s_x, U=s_u, λ=s_λ, μ=s_μ, ν=s_ν, s=s_s)

    function newton_cost(V::Vector)
        X,U,λ,μ,ν,s = unpackV(V)
        J = cost(solver,X,U) + λ'cE(V) + μ'*(cI(V) + 0.5*s.^2) + ν'd(V)
        return J
    end

    function cI(V)
        X,U,λ,μ,ν,s = unpackV(V)
        C = zeros(eltype(V),pI,N-1)
        for k = 1:N-1
            cI!(view(C,1:pI,k),X[:,k],U[:,k])
        end
        CN = zeros(eltype(V),pI_N)
        cI!(CN,X[:,N])
        CI = [vec(C); CN]
        return CI
    end

    function cE(V)
        X,U,λ,μ,ν,s = unpackV(V)
        C = zeros(eltype(V),pE,N-1)
        for k = 1:N-1
            cE!(view(C,1:pE,k),X[:,k],U[:,k])
        end
        CN = zeros(eltype(V),pE_N)
        cE!(CN,X[:,N])
        CE = [vec(C); CN]
        return CE
    end

    function d(V)
        X,U,λ,μ,ν,s = unpackV(V)
        D = zeros(n,N)
        D[:,1] = X[:,1] - obj.x0
        for k = 2:N
            f(view(D,1:n,k),X[:,k-1],U[:,k-1])
            D[:,k] -= X[:,k]
        end
        return vec(D)
    end

    function packV(X,U,λ,μ,ν,s)
        V = [vec(X);vec(U);λ;μ;ν;s]
    end

    function unpackV(V)
        X = V[sze.X]
        U = V[sze.U]
        λ = V[sze.λ]
        μ = V[sze.μ]
        ν = V[sze.ν]
        s = V[sze.s]
        return X,U,λ,μ,ν,s
    end
    return newton_cost, packV, unpackV, cI, cE, d
end

model,obj = Dynamics.dubinscar_parallelpark
solver = Solver(model,obj,N=11)
n,m,N = get_sizes(solver)

newton_cost, packV, unpackV, cI, cE, d = gen_newton_functions(solver)
p,pI,pE = get_num_constraints(solver)
pN,pI_N,pE_N = get_num_terminal_constraints(solver)
X = rand(n,N)
U = rand(m,N-1)
λ = rand(pE*(N-1)+pE_N)
μ = rand(pI*(N-1)+pI_N)
ν = rand(N*n)
s = rand(size(μ)...)

V = packV(X,U,λ,μ,ν,s)
(X,U,λ,μ,ν,s) == unpackV(V)

newton_cost(V)
cost(solver,X,U) +μ'*(cI(V) + 0.5*s.^2) + ν'd(V)

X = rollout(solver,U)
V = packV(X,U,λ,μ,ν,s)
d(V)
X[:,N]

cE(V) == X[:,N] - obj.xf
cI(V)[end-3:end]
