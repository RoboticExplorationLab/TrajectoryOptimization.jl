using ForwardDiff
using LinearAlgebra
using Plots

model,obj = Dynamics.pendulum
obj.cost.Q .= Diagonal(I,2)
# obj = ConstrainedObjective(obj)x

# obj_c = ConstrainedObjective(obj,u_min=-0.3,u_max=0.4)
solver = Solver(model,obj,N=21)
n,m,N = get_sizes(solver)
dt = solver.dt
U0 = ones(m,N-1)
solver.opts.verbose = true
res,stats = solve(solver,U0)
plot(res.X)


function mycost(Z)
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx+1:end],m,N-1)
    cost(solver,X,U)
end

function lagrangian(V)
    nu = V[Nz+1:end]
    Z = V[1:Nz]
    J = mycost(Z) + nu'dynamics(Z)
end

function al_lagrangian(V,ρ)
    Z = V[1:Nz]
    d = dynamics(Z)
    lagrangian(V) + ρ/2*(d'd)
end


function dynamics(Z)
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx.+(1:Nu)],m,N-1)

    D = zeros(eltype(Z),n,N)
    D[:,1] = X[:,1] - solver.obj.x0
    for k = 2:N
        solver.fd(view(D,1:n,k),X[:,k-1],U[:,k-1])
        D[:,k] -= X[:,k]
    end
    return vec(D)
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

function solve_newton(V,ρ,type; iters=10, verbose=false, iters_linesearch=10)
    # Define merit function
    meritfun(V) = al_lagrangian(V,ρ)
    max_c(V) = norm(dynamics(V),Inf)

    # Initial cost
    J0 = meritfun(V)
    V_ = copy(V)
    println("Initial Cost: $J0")

    # Stats
    cost = zeros(iters)
    grad = zeros(iters)
    c_max = zeros(iters)

    # Take Newton Steps
    for i = 1:iters
        A,b = buildKKT(V_,ρ,type)
        δV = -A\b
        α = armijo_line_search(meritfun,V_,δV,b, max_iter=iters_linesearch)
        V_ = V_ + α*δV
        J = meritfun(V_)
        cost[i] = J
        grad[i] = norm(b)
        c_max[i] = max_c(V_)
        if verbose
            println("Iter $i:")
            println("  cost: $J")
            println("  grad: $(grad[i])")
            println("  c_max: $(c_max[i])")
            println("  α: $α")
            println("  rank: $(rank(A))")
            println("  cond: $(cond(A))")
        end
    end
    stats = Dict("cost"=>cost,"grad"=>grad,"c_max"=>c_max)
    return V, stats
end

function buildKKT(V,ρ,type=:penalty)
    Z = V[1:Nz]
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx+1:end],m,N-1)
    nu = V[Nz+1:end]

    ∇²J = ForwardDiff.hessian(mycost,Z)
    ∇J = ForwardDiff.gradient(mycost,Z)
    D = ForwardDiff.jacobian(dynamics,Z)
    d = dynamics(Z)

    if type == :penalty
        A = [∇²J   D';
             D   -1/ρ*I]
        b = [∇J + D'nu; d]
    elseif type == :kkt
        A = [∇²J   D';
             D   zeros(Nx,Nx)]
        b = [∇J + D'nu; d]
    elseif type == :ad_lagrangian
        A = ForwardDiff.hessian(lagrangian,V)
        b = ForwardDiff.gradient(lagrangian,V)
    elseif type == :ad_aulag
        meritfun(V) = al_lagrangian(V,ρ)
        A = ForwardDiff.hessian(meritfun,V)
        b = ForwardDiff.gradient(meritfun,V)
    elseif type == :gradient_descent
        A = Diagonal(I,NN)
        b = ForwardDiff.gradient(lagrangian,V)
    end
    return A,b
end

Nx = N*n
Nu = (N-1)*m
Nz = Nx + Nu
NN = 2Nx + Nu
x = vec(res.X)
u = vec(res.U)
nu = zeros(Nx)

x = vec(rollout(solver,U0))
u = vec(U0)


# Build KKT System
ρ = 0
Z = [x;u]
V = [x;u;nu]

J0 = al_lagrangian(V,ρ)
dynamics(V)

A,b = buildKKT(V,ρ,:penalty)
A_,b_ = buildKKT(V,ρ,:kkt)
dv = -A_\b_
α = armijo_line_search(lagrangian,V,dv,b_)

V1 = V + dv*α
dynamics(V1)
lagrangian(V1)


ForwardDiff.gradient(lagrangian,V1)
Z = V1[1:Nz]
X = reshape(Z[1:Nx],n,N)
U = reshape(Z[Nx.+(1:Nu)],m,N-1)
plot(X')
lagrangian(V1)

Afd,bfd = buildKKT(V,ρ,:ad_lagrangian)

meritfun(V) = al_lagrangian(V,ρ)
armijo_line_search(meritfun,V,-b,b; max_iter=15)
J0 = meritfun(V)
meritfun(V + 0.001*b)

# Check difference in Hessians
norm(Afd-A_)
norm(bfd-b_)



ρ = 1e8
V_pen, stats_pen = solve_newton(V,ρ,:penalty,verbose=true)
V_kkt, stats_kkt = solve_newton(V,ρ,:kkt,verbose=true)
V_fd, stats_fd = solve_newton(V,ρ,:ad_lagrangian,verbose=true)
V_fda, stats_fda = solve_newton(V,ρ,:ad_aulag)
V_grd, stats_grd = solve_newton(V,ρ,:gradient_descent,iters_linesearch=40)

mycost(V_pen[1:Nz])
val = "c_max"
plot(stats_pen[val],label="penalty",title=val,xlabel="iteration",yscale=:log10)
plot!(stats_kkt[val],label="kkt")
plot!(stats_fd[val],label="fd-lagrangian")
plot!(stats_fda[val],label="fd-aug lagrangian")
plot!(stats_grd[val],label="gradient descent")
