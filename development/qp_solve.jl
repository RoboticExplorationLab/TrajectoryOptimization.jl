function gen_qp_functions(solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    c_function!, c_jacobian!, c_labels, cI!, cE! = generate_constraint_functions(solver.obj)
    f = solver.fd
    s_x = reshape(1:(N*n),n,N)
    s_u = reshape(N*n .+ (1:(m*(N-1))),m,N-1)
    s_ν = (N*n + (N-1)*m) .+ (1:N*n)
    sze = (X=s_x, U=s_u, ν=s_ν)

    function newton_cost(V::Vector)
        X,U = unpackV(V)
        J = cost(solver,X,U)
        return J
    end

    function cI(V)
        X,U = unpackV(V)
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
        X,U = unpackV(V)
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
        X,U = unpackV(V)
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


    function packV(X,U)
        V = [vec(X);vec(U)]
    end

    function unpackV(V)
        X = V[sze.X]
        U = V[sze.U]
        return X,U,λ,μ,ν,s
    end
    return newton_cost, packV, unpackV, cI, cE, d, active_set, sze
end

model,obj = Dynamics.dubinscar
solver = Solver(model,obj,N=31)
n,m,N = get_sizes(solver)
U0 = rand(m,N-1)
res,stats = solve(solver,U0)
# λ_update_default!(res,solver);

newton_cost, packV, unpackV, cI, cE, d, active_set = gen_qp_functions(solver)
X = to_array(res.X)
U = to_array(res.U)
ν = vec(to_array(res.s))
V = packV(X,U)
Nz = length(V)


using JuMP, NLopt
using Gurobi
H = ForwardDiff.hessian(newton_cost,V)
g = ForwardDiff.gradient(newton_cost,V)
c = newton_cost(V)
function quad_cost(δV)
    newton_cost(V) + g'δV + 0.5*δV'H*δV
end

∇d = ForwardDiff.jacobian(d,V)
e = d(V)
d(V)
function quad_dynamics(δV)
    d(V) + ∇d*δV
end

function qp_step(V)
    H = ForwardDiff.hessian(newton_cost,V)
    g = ForwardDiff.gradient(newton_cost,V)
    c = newton_cost(V)

    ∇d = ForwardDiff.jacobian(d,V)
    e = d(V)

    mod = JuMP.Model(solver=GurobiSolver())
    @variable(mod, Z[1:Nz])
    obj = 0.5*Z'H*Z + g'Z + c
    @objective(mod, Min, obj)
    dyn = e + ∇d*Z;
    @constraint(mod, dyn .== 0)
    status = JuMP.solve(mod)
    return getvalue(Z)
end

V = packV(X,U)
newton_cost(V)
norm(d(V),Inf)

max_d = Float64[]

for i = 1:20
    V_ = qp_step(V) + V
    newton_cost(V_)
    push!(max_d,norm(d(V_),Inf))
    copyto!(V,V_)
end

plot(max_d)
