using ForwardDiff
using LinearAlgebra
using Plots
import TrajectoryOptimization: get_num_terminal_constraints, generate_constraint_functions

model,obj = Dynamics.pendulum
obj.cost.Q .= Diagonal(I,2)*1
obj.cost.R .= Diagonal(I,1)*1
obj = ConstrainedObjective(obj,u_max=3)
obj = update_objective(obj,tf=5)
obj.cost.Qf .= Diagonal(I,2)*1

model,obj = Dynamics.dubinscar
obj = ConstrainedObjective(obj,u_max=0.75,u_min=-0.75)#,x_min=[-0.5;-0.01;-Inf])

# obj_c = ConstrainedObjective(obj,u_min=-0.3,u_max=0.4)
solver = Solver(model,obj,N=11)
n,m,N = get_sizes(solver)
p,pI,pE = get_num_constraints(solver)
p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
c_function!, c_jacobian!, c_labels, cI!, cE! = generate_constraint_functions(solver.obj)
dt = solver.dt
U0 = ones(m,N-1)
solver.opts.verbose = true
solver.opts.penalty_initial = 0.01
solver.opts.cost_tolerance_intermediate = 1e-3
res,stats = solve(solver,U0)
plot()
plot_trajectory!(res)
plot(res.U)


function mycost(Z)
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx+1:end],m,N-1)
    cost(solver,X,U)

end

function lagrangian(V)
    nu = V[Nz .+ (1:Nx)]
    λ = V[(Nz + Nx) .+ (1:Nh)]
    μ = V[(Nz + Nx + Nh) .+ (1:Ng)]
    Z = V[1:Nz]
    J = mycost(Z) + nu'dynamics(Z) + λ'cE(Z) + μ'cI(Z)
end

function al_lagrangian(V,ρ)
    Z = V[1:Nz]
    d = dynamics(Z)
    h = cE(Z)
    g = cI(Z)
    lagrangian(V) + ρ/2*(d'd + h'h + g'g)
end

function cE(Z)
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx.+(1:Nu)],m,N-1)

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
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx.+(1:Nu)],m,N-1)

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

function active_set(Z,eps=0)
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx.+(1:Nu)],m,N-1)

    a = ones(Bool,NN)
    ci = cI(Z)
    c_inds = ones(Bool,length(ci))
    c_inds = ci .>= -eps

    a[(Nz + Nx + Nh) .+ (1:Ng)] = c_inds
    # a[ind1.s] = c_inds
    return a
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
    nu = V[Nz.+(1:Nx)]
    λ = V[(Nz + Nx) .+ (1:Nh)]
    μ = V[(Nz + Nx + Nh) .+ (1:Ng)]

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
        b = [∇J + D'nu + H'λ + G'μ; d; h; g]
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

function viewmatrix(A::Matrix,filename="logs/viewer.txt")
    io = open(filename,"w")
    show(IOContext(io, :limit=>false),"text/plain",A)
    close(io)
end

isfullrank(A) = rank(A) == minimum(size(A))


Nx = N*n
Nu = (N-1)*m
Nz = Nx + Nu
Nh = (N-1)*pE + pE_N
Ng = (N-1)*pI + pI_N
NN = 2Nx + Nu + Nh + Ng
names = (:z,:ν,:λ,:μ)
ind1 = TrajectoryOptimization.create_partition([Nz,Nx,Nh,Ng],names)
ind2 = TrajectoryOptimization.create_partition2([Nz,Nx,Nh,Ng],names)

x = vec(res.X)
u = vec(res.U)
nu = zeros(Nx)

s = zeros(pI,N-1)
μ = zeros(pI,N-1)
λ = zeros(pE,N-1)

ineq = 1:pI
eq = pI .+ (1:pE)
for k = 1:N-1
    s[:,k] = sqrt.(2.0*max.(0,-res.C[k][ineq]))
    μ[:,k] = res.λ[k][ineq]
    λ[:,k] = res.λ[k][eq]
end
s = vec(s)
μ = vec(μ)
λ = vec(λ)
append!(s, sqrt.(2.0*max.(0,-res.C[N][1:pI_N])))
append!(μ, res.λ[N][1:pI_N])
append!(λ, res.λ[N][pI_N .+ (1:pE_N)])

inds = (z=1:Nz, ν=Nz.+(1:Nx), λ=(Nz+Nx) .+ (1:Nh), μ=(Nz+Nx+Nh) .+ (1:Ng))

# Build KKT System
ρ = 100
Z = [x;u]
V = [x;u; nu; λ; μ]

J0 = al_lagrangian(V,ρ)
dynamics(V)
g0 = cI(V)

# Build KKT
A,b = buildKKT(V,ρ,:penalty)
A_,b_ = buildKKT(V,ρ,:kkt)
a = active_set(Z,1e-2)
n_active = Ng - (length(a) - count(a))
amu = a[inds.μ]

D = ForwardDiff.jacobian(dynamics,Z)
H = ForwardDiff.jacobian(cE,Z)
G = ForwardDiff.jacobian(cI,Z)
isfullrank(D)
isfullrank(H)
isfullrank(G[amu,:])

# Take Step
dv = zero(V)
dv[a] = -A_[a,a]\b_[a]
A2 = A_[a,a]
Ga = A2[ind1.z,Nz+Nx+Nh+1:end]
Ga == G[amu,:]'
isfullrank(A_[ind2.zμ])
rank(A_[a,a])
rank(A)
cond(A)
# dv[a] = -A[a,a]\b[a]

α = armijo_line_search(lagrangian,V,dv,b_)
V1 = V + dv*α

# Evaluate
dJ = J0 - lagrangian(V1)
norm(dynamics(V1))
norm(dynamics(V1),Inf)
argmax(abs.(dynamics(V1)))
norm(cE(V1))
norm(b_)

g = cI(V1)
maximum(g[amu])
maximum(g)
amu[argmax(g)] == false

mu1 = V1[inds.μ]
mu1[amu]

# Cap multipier
mu1 .= max.(0,mu1)
V1[inds.μ] = mu1
V = copy(V1)




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



ρ = 1e10
V_pen, stats_pen = solve_newton(V,ρ,:penalty,verbose=true)
V_kkt, stats_kkt = solve_newton(V,ρ,:kkt,verbose=true)
V_fd, stats_fd = solve_newton(V,ρ,:ad_lagrangian,verbose=true)
V_fda, stats_fda = solve_newton(V,ρ,:ad_aulag)


mycost(V_pen[1:Nz])
val = "c_max"
plot(stats_pen[val],label="penalty",title=val,xlabel="iteration",yscale=:log10)
plot!(stats_kkt[val],label="kkt")
plot!(stats_fd[val],label="fd-lagrangian")
plot!(stats_fda[val],label="fd-aug lagrangian")
