function gen_newton_al_functions(solver::Solver)
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

    function newton_cost(V::Vector,ρ=1)
        X,U,λ,μ,ν,s = unpackV(V)
        e = cE(V)
        i = cI(V) + 0.5s.^2
        dd = d(V)
        J = cost(solver,X,U) + λ'e + μ'i + ν'dd + 0.5*ρ*(e'e + i'i + dd'dd)
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
        D = zeros(eltype(V),n,N)
        D[:,1] = X[:,1] - solver.obj.x0
        for k = 2:N
            f(view(D,1:n,k),X[:,k-1],U[:,k-1])
            D[:,k] -= X[:,k]
        end
        return vec(D)
    end

    function active_set(V,eps=1e-2)
        X,U,λ,μ,ν,s = unpackV(V)
        a = trues(length(V))
        ci = cI(V)
        c_inds = ci .>= -eps
        a[sze.μ] = c_inds
        a[sze.s] = c_inds
        return a
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
    return newton_cost, packV, unpackV, cI, cE, d, active_set, sze
end

function init_newton_results(results::SolverIterResults,Solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    pN,pI_N,pE_N = get_num_terminal_constraints(solver)

    X = to_array(res.X)
    U = to_array(res.U)
    μ = vcat([k < N ? res.λ[k][1:pI] : res.λ[k][1:pI_N] for k = 1:N]...)
    λ = vcat([k < N ? res.λ[k][pI .+ (1:pE)] : res.λ[k][pI_N .+ (1:pE_N)] for k = 1:N]...)
    ν = vec(to_array(res.s))
    s = vcat([k < N ? res.C[k][1:pI] : res.C[k][1:pI_N] for k = 1:N]...)
    s = sqrt.(2*max.(0,-s))
    packV(X,U,λ,μ,ν,s)
end

function line_search(V,H,g,a=trues(length(V)))
    J0 = mycost(V)
    α = 1.
    V_ = copy(V)
    delta = (H[a,a]\g[a])
    V_[a] = V[a] - α*delta
    J = mycost(V_)
    iter = 1
    while J > J0
        α /= 2
        V_[a] = V[a] - α*delta
        J = mycost(V_)
        iter += 1
        @show J, α
        iter > 10 ? break : nothing
    end
    @show J0-J
    return V_
end


# Block move
model, obj = TrajectoryOptimization.Dynamics.double_integrator
u_min = -0.2
u_max = 0.2
x_bnd = 2
obj_con = TrajectoryOptimization.ConstrainedObjective(obj, tf=5.0,use_xf_equality_constraint=true,u_min=u_min, u_max=u_max, x_max=x_bnd, x_min=-x_bnd)#, x_max=x_max)

solver = TrajectoryOptimization.Solver(model,obj_con,integration=:rk4,N=5,opts=opts)
Random.seed!(1)
U = rand(solver.model.m, solver.N)

res, stats = TrajectoryOptimization.solve(solver,U)
@assert max_violation(res) < opts.constraint_tolerance

newton_cost, packV, unpackV, cI, cE, d, active_set = gen_newton_al_functions(solver)
max_c(V) = max(maximum(cI(V)),norm(cE(V),Inf),norm(d(V),Inf))
V = init_newton_results(res,solver)

cost(solver,res)
newton_cost(V)
max_violation(res)
max_c(V)
g = ForwardDiff.gradient(newton_cost,V)
H = ForwardDiff.hessian(newton_cost,V)
a = active_set(V,1.0)
rank(H[a,a])
size(H[a,a])
V[a] = V[a] - (H[a,a] + 0I)\g[a]
# V = V - H\g
newton_cost(V)
max_c(V)


# dubinscar
model,obj = Dynamics.dubinscar_parallelpark
solver = Solver(model,obj,N=21)
solver.opts.cost_tolerance = 1e-4
solver.opts.constraint_tolerance = 1e-3
solver.opts.penalty_max = 1e4
n,m,N = get_sizes(solver)
Random.seed!(1)
U0 = rand(m,N-1)
res,stats = solve(solver,U0)
cost(solver,res)
λ_update_default!(res,solver);
update_constraints!(res,solver)
@assert max_violation(res) < solver.opts.constraint_tolerance
ρ = maximum(maximum.(res.μ))
maximum(vcat(res.C...) .* vcat(res.active_set...))

newton_cost, packV, unpackV, cI, cE, d, active_set = gen_newton_al_functions(solver)
mycost(V) = newton_cost(V,1e4)
max_c(V) = max(maximum(cI(V)),norm(cE(V),Inf),norm(d(V),Inf))
c_all = [cI(V); abs.(cE(V)); abs.(d(V))]
argmax(c_all)
V = init_newton_results(res,solver)

cost(solver,res)
mycost(V)
max_violation(res)
max_c(V)
g = ForwardDiff.gradient(mycost,V)
H = ForwardDiff.hessian(mycost,V)
a = active_set(V,1e-3)
rank(H)
rank(H[a,a])
size(H[a,a])
# V[a] - H[a,a]\g[a]
V1 = line_search(V,H+1000I*0,g,a)
# V = V - H\g
mycost(V1)
max_c(V1)
ForwardDiff.gradient(mycost,V1)
copyto!(V,V1)
max_c(V1)

J = Float64[]
c_max = Float64[]
for i = 1:10
